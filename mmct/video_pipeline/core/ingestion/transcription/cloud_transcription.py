import asyncio
import os
import aiofiles
import json
from typing import List, Optional, Dict, Any
from loguru import logger
from mmct.video_pipeline.core.ingestion.models import TranslationResponse
from mmct.video_pipeline.core.ingestion.transcription.base_transcription import (
    Transcription,
)
from mmct.video_pipeline.utils.helper import extract_wav_from_video
from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.providers.factory import provider_factory
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class CloudTranscription(Transcription):
    def __init__(self, video_path: str, hash_id: str, language: str = None) -> None:
        super().__init__(video_path=video_path, hash_id=hash_id, language=language)
        self.audio_container = os.getenv("AUDIO_CONTAINER_NAME")
        self.local_save = []
        # Initialize providers
        self.llm_provider = provider_factory.create_llm_provider()
        self.speech_provider = provider_factory.create_transcription_provider('azure_speech')

    async def _load_audio(self):
        try:
            logger.info(f"Extracting the audio from the video: {self.video_path}")
            self.audio_path = os.path.join(await get_media_folder(), f"{self.hash_id}.wav")
            logger.info(f"Target audio path: {self.audio_path}")

            # Verify video file exists before extraction
            if not os.path.exists(self.video_path):
                raise FileNotFoundError(f"Video file not found for audio extraction: {self.video_path}")

            await extract_wav_from_video(video_path=self.video_path, output_path=self.audio_path)

            # Verify audio file was created successfully
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio extraction failed - output file not created: {self.audio_path}")

            audio_size = os.path.getsize(self.audio_path)
            logger.info(f"Audio extracted successfully: {self.audio_path} (size: {audio_size} bytes)")

            self.local_save.append(self.audio_path)
            return "", self.local_save
        except Exception as e:
            logger.exception(f"Error loading audio, {e}")
            raise

    async def detect_language(self):
        try:
            languages = ["en-IN", "hi-IN", "te-IN", "or-IN"]
            detected_language = await self.speech_provider.detect_language(
                audio_path=self.audio_path,
                candidate_languages=languages
            )
            return detected_language
        except Exception as e:
            logger.exception(f"Error while detection language, Error:{e}")
            raise

    async def get_transcript(self):
        try:
            if self.source_language["lang-code"] is None:
                lang = await self.detect_language()
                self.source_language["lang-code"] = (
                    lang if lang not in [None, "Unknown"] else "en-IN"
                )
                self.source_language["lang"] = Languages(self.source_language["lang-code"]).name

            # Check if audio file exists
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

            file_size = os.path.getsize(self.audio_path)
            logger.info(f"Audio file path: {self.audio_path}, size: {file_size} bytes")

            # Prepare phrase list for Hindi
            phrase_list = None
            if self.source_language["lang-code"] == "hi-IN":
                phrase_list = self.hindi_glossary

            # Use speech provider for transcription
            result = await self.speech_provider.transcribe_file(
                audio_path=self.audio_path,
                language=self.source_language["lang-code"],
                phrase_list=phrase_list
            )

            logger.info(f"Transcription completed with {len(result)} segments")
            if not result:
                logger.warning("No transcription results obtained!")

            return result

        except Exception as e:
            logger.exception(f"Azure Transcription failed, Error: {e}")
            raise

    async def _get_formatted_transcript(self, transcript: List[Dict[str, Any]]) -> Optional[str]:
        try:
            logger.info("Formatting the generated transcript..")
            if not isinstance(transcript, list):
                return None

            formatted_transcript = ""
            for idx, segment in enumerate(transcript, 1):
                if not segment.get("text"):
                    continue
                formatted_transcript += (
                    f"{idx}\n"
                    f"{segment['start_time']},000 --> {segment['end_time']},000\n"
                    f"{segment['text']}\n\n"
                )
            logger.info("Successfully formatted the transcript")
            return formatted_transcript
        except Exception as e:
            logger.exception(f"Formatting failed, Error: {e}")
            raise

    async def _translate_batch(self, batch, max_retries=3, current_retry=0, prompt=None):
        """Translate a batch of text with retry logic for handling response mismatches"""
        to_translate = json.dumps([e["text"] for e in batch], ensure_ascii=False)
        logger.info(
            f"Translating batch of {len(batch)} entries (retry {current_retry}/{max_retries}):\n{to_translate}"
        )
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
            {"role": "user", "content": f"Text to translate:\n{to_translate}"},
        ]

        result = await self.llm_provider.chat_completion(
            messages=messages,
            temperature=0,
            top_p=0.1,
            response_format=TranslationResponse,
        )

        translation_response: TranslationResponse = result['content']
        translations = translation_response.translations

        # Check if number of translations matches number of entries
        if len(translations) != len(batch):
            logger.warning(
                f"Mismatch: received {len(translations)} translations for {len(batch)} entries"
            )

            # If we've reached max retries or have only one entry, return what we have
            if current_retry >= max_retries or len(batch) <= 1:
                raise ValueError("Max retries reached for translation, or can't split further.")

            # Split the batch and retry with smaller batches
            mid = len(batch) // 2
            first_half = await self._translate_batch(batch[:mid], max_retries, current_retry + 1)
            second_half = await self._translate_batch(batch[mid:], max_retries, current_retry + 1)
            return first_half + second_half

        return translations

    async def _get_translated_transcript(
        self, text: str, max_chars_per_batch: int = 2000
    ) -> TranslationResponse:
        try:
            logger.info("Retrieving the translated transcript")
            raw_blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
            entries = []  # type: List[dict]
            logger.info("Splitting the texts into chunks for translation")
            for block in raw_blocks:
                lines = block.splitlines()
                if len(lines) < 3:
                    continue
                seq_no = lines[0]
                timestamp = lines[1]
                content = "\n".join(lines[2:])
                entries.append({"seq": seq_no, "time": timestamp, "text": content})

            # Batch entries by text length
            batches: List[List[dict]] = []
            curr_batch, curr_len = [], 0
            logger.info("Aggregating the text chunks into batches")
            for entry in entries:
                length = len(entry["text"])
                if curr_len + length > max_chars_per_batch and curr_batch:
                    batches.append(curr_batch)
                    curr_batch, curr_len = [], 0
                curr_batch.append(entry)
                curr_len += length
            if curr_batch:
                batches.append(curr_batch)

            prompt = """You are a highly skilled translator. Your task is to translate the provided JSON array of text from {source_language} to English with utmost accuracy.

            # Instructions:
            - Translate each line of the input text exactly as it is, without adding, omitting, or altering any information.
            - The input text may include different dialects of {source_language}; translate them carefully while preserving the original meaning.
            - Do not hallucinate or introduce any new information that is not present in the input text.
            - If a term or phrase is unclear, translate it as closely as possible to its original meaning without making assumptions.
            """
            prompt = prompt.format(
                source_language=self.source_language["lang"].split("_")[0].capitalize()
            )
            logger.info("Inserting the source language to prompt")
            if self.source_language["lang-code"] == "hi-IN":
                logger.info("Adding glossary for hindi vocabulary")
                glossary_table = self.glossary_df[
                    self.glossary_df["hindi_terms"].apply(lambda term: term in text)
                ].to_markdown(index=False)
                prompt += f"\n\n# Glossary:\n{glossary_table}\n"

            all_translations: List[str] = []
            logger.info("Translating the texts batchwise")
            # Process each batch with retry logic
            for batch in batches:
                batch_translations = await self._translate_batch(batch, prompt=prompt)
                all_translations.extend(batch_translations)

            # Reassemble into SRT format
            output_blocks = []
            for entry, translation in zip(entries, all_translations):
                block = "\n".join([entry["seq"], entry["time"], translation.strip()])
                output_blocks.append(block)

            return "\n\n".join(output_blocks)
        except Exception as e:
            logger.exception(f"Error translating transcript: {e}")
            raise

    async def __call__(self):
        try:
            _, self.local_save = await self._load_audio()
            transcript = await self.get_transcript()  # Speech to text
            logger.info(f"transcript created via azure-stt:{transcript}")
            transcript = await self._get_formatted_transcript(
                transcript=transcript
            )  # Formatting the transcript as same as whisper
            logger.info(f"formatted transcript:{transcript}")
            if self.source_language["lang-code"] != "en-IN":
                transcript = await self._get_translated_transcript(
                    transcript
                )  # translating to english
                logger.info(f"translated {transcript}")
            transcript_save_path = os.path.join(
                await get_media_folder(), f"transcript_{self.hash_id}.srt"
            )
            async with aiofiles.open(transcript_save_path, "w", encoding="utf-8") as f:
                await f.write(transcript)

            self.local_save.append(transcript_save_path)

            return transcript, self.local_save
        except Exception as e:
            logger.exception(f"Exception occured : {e}")
            raise


if __name__ == "__main__":
    # Example usage - replace with your actual values
    video_path = "path/to/your/video.mp4"
    hash_id = "example_hash_id"
    transcriber = CloudTranscription(video_path=video_path, hash_id=hash_id)
    transcript = asyncio.run(transcriber.run())
    print("Final Transcript:")
    print(transcript)
