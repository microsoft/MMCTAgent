import asyncio
import time
import os
import aiofiles
import json
from typing import List, Optional, Dict, Any
from loguru import logger
from mmct.video_pipeline.core.ingestion.models import TranslationResponse
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential, AzureCliCredential
from mmct.video_pipeline.core.ingestion.transcription.base_transcription import (
    Transcription,
)
from mmct.video_pipeline.utils.helper import extract_wav_from_video
from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import get_media_folder
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


class CloudTranscription(Transcription):
    def __init__(self, video_path: str, hash_id: str, language: str = None) -> None:
        super().__init__(video_path=video_path, hash_id=hash_id, language=language)
        self.audio_container = os.getenv("AUDIO_CONTAINER_NAME")
        self.local_save = []

    async def _load_audio(self):
        try:
            logger.info("Extracting the audio from the video")
            self.audio_path = os.path.join(
                await get_media_folder(), f"{self.hash_id}.wav"
            )

            await extract_wav_from_video(
                video_path=self.video_path, output_path=self.audio_path
            )
            self.local_save.append(self.audio_path)
        except Exception as e:
            logger.exception(f"Error loading audio, {e}")
            raise

    async def detect_language(self):
        try:
            # Use Azure CLI credential first, then fallback to DefaultAzureCredential
            credential = await self._get_credential()
                
            token = credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
            token = token.token
            resource_id = os.getenv("SPEECH_SERVICE_RESOURCE_ID")
            token = "aad#" + resource_id + "#" + token
            speech_config = speechsdk.SpeechConfig(
                region=os.getenv("SPEECH_SERVICE_REGION"), auth_token=token
            )
            audio_config = speechsdk.audio.AudioConfig(filename=self.audio_path)
            lang = None
            conf = None
            languages = ["en-IN", "hi-IN", "te-IN", "or-IN"]
            auto_detect_source_language_config = (
                speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=languages
                )
            )
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect_source_language_config,
                audio_config=audio_config,
            )
            result = speech_recognizer.recognize_once()
            auto_detect_source_language_result = (
                speechsdk.AutoDetectSourceLanguageResult(result)
            )
            detected_language = auto_detect_source_language_result.language
            # if len(result.text)!=0:
            #     json_result = result.properties[speechsdk.PropertyId.SpeechServiceResponse_JsonResult]
            #     json_result = ast.literal_eval(json_result)
            #     lang = json_result['PrimaryLanguage']['Language']
            #     conf = json_result['primaryLanguage']['Confidence']
            # else:
            #     print("Unable to detect")
            # if lang and conf and conf=="High":
            #     return lang
            return detected_language
        except Exception as e:
            logger.exception(f"Error while detection language, Error:{e}")
            raise

    async def _get_credential(self):
        """Get Azure credential, trying CLI first, then DefaultAzureCredential."""
        try:
            # Try Azure CLI credential first
            cli_credential = AzureCliCredential()
            # Test if CLI credential works by getting a token
            cli_credential.get_token("https://cognitiveservices.azure.com/.default")
            return cli_credential
        except Exception:
            return DefaultAzureCredential()

    async def get_transcript(self):
        try:
            result = []
            # Use Azure CLI credential first, then fallback to DefaultAzureCredential
            credential = await self._get_credential()
                

            token = credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
            token = token.token
            resource_id = os.getenv("SPEECH_SERVICE_RESOURCE_ID")
            auth_token = f"aad#{resource_id}#{token}"
            speech_config = speechsdk.SpeechConfig(
                region=os.getenv("SPEECH_SERVICE_REGION"), auth_token=auth_token
            )
            logger.info(f"Speech Config initialized with region: {os.getenv('SPEECH_SERVICE_REGION')}")
            logger.info(f"Using resource ID: {resource_id}")
            if self.source_language == None:
                lang = await self.detect_language()
                self.source_language["lang-code"] = (
                    lang if lang not in [None, "Unknown"] else "en-IN"
                )
                self.source_language["lang"] = Languages(
                    self.source_language["lang-code"]
                ).name
            speech_config.speech_recognition_language = self.source_language[
                "lang-code"
            ]
            # Check if audio file exists
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            
            file_size = os.path.getsize(self.audio_path)
            logger.info(f"Audio file path: {self.audio_path}, size: {file_size} bytes")
            
            audio_config = speechsdk.audio.AudioConfig(filename=self.audio_path)

            transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config, audio_config=audio_config
            )

            # 3) (Optional) Add your phrase list -- Only available for Hindi right now
            if self.source_language["lang-code"] == "hi-IN":
                phrase_list = speechsdk.PhraseListGrammar.from_recognizer(transcriber)
                for phrase in self.hindi_glossary[:500]:
                    phrase_list.addPhrase(phrase)

            # 4) Prepare an asyncio.Event + thread‑safe setter
            loop = asyncio.get_running_loop()
            done_evt = asyncio.Event()

            def _stop_cb(evt: speechsdk.SessionEventArgs):
                loop.call_soon_threadsafe(done_evt.set)

            # 5) Define all callbacks
            def _on_session_started(evt):
                logger.info("Session started")

            def _on_session_stopped(evt):
                logger.info("Session stopped")
                _stop_cb(evt)

            def _on_canceled(evt):
                logger.info("Canceled")
                _stop_cb(evt)

            def _on_transcribed(evt: speechsdk.SpeechRecognitionEventArgs):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    start_ms = evt.result.offset
                    dur_ms = evt.result.duration
                    # convert from 100‑nanos to seconds
                    start_s = start_ms / 10_000_000
                    end_s = (start_ms + dur_ms) / 10_000_000
                    rec = {
                        "text": evt.result.text,
                        "start_time": time.strftime("%H:%M:%S", time.gmtime(start_s)),
                        "end_time": time.strftime("%H:%M:%S", time.gmtime(end_s)),
                        "speaker_id": evt.result.speaker_id,
                    }
                    logger.info(
                        f"Transcribed: {rec['text']} [{rec['start_time']}–{rec['end_time']}]"
                    )
                    result.append(rec)
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.error(f"NoMatch: {evt.result.no_match_details}")

            # 6) Wire up callbacks
            transcriber.session_started.connect(_on_session_started)
            transcriber.session_stopped.connect(_on_session_stopped)
            transcriber.canceled.connect(_on_canceled)
            transcriber.transcribed.connect(_on_transcribed)

            # 7) Kick off transcription
            start_future = transcriber.start_transcribing_async()
            start_future.get()  # wait for the SDK to begin

            # 8) Wait until the SDK signals stop/cancel
            await done_evt.wait()

            # 9) Clean shutdown
            stop_future = transcriber.stop_transcribing_async()
            stop_future.get()
            
            logger.info(f"Transcription completed with {len(result)} segments")
            if not result:
                logger.warning("No transcription results obtained!")
            
            return result

        except Exception as e:
            logger.exception(f"Azure Transcription failed, Error: {e}")
            raise

    async def _get_formatted_transcript(
        self, transcript: List[Dict[str, Any]]
    ) -> Optional[str]:
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

    async def _translate_batch(
        self, batch, max_retries=3, current_retry=0, prompt=None
    ):
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

        response = await self.llm_client.beta.chat.completions.parse(
            model=os.getenv(
                "LLM_MODEL_NAME"
                if os.getenv("LLM_PROVIDER") == "azure"
                else "OPENAI_MODEL_NAME"
            ),
            messages=messages,
            temperature=0,
            top_p=0.1,
            response_format=TranslationResponse,
        )

        translation_response: TranslationResponse = response.choices[0].message.parsed
        translations = translation_response.translations

        # Check if number of translations matches number of entries
        if len(translations) != len(batch):
            logger.warning(
                f"Mismatch: received {len(translations)} translations for {len(batch)} entries"
            )

            # If we've reached max retries or have only one entry, return what we have
            if current_retry >= max_retries or len(batch) <= 1:
                raise ValueError(
                    "Max retries reached for translation, or can't split further."
                )

            # Split the batch and retry with smaller batches
            mid = len(batch) // 2
            first_half = await self._translate_batch(
                batch[:mid], max_retries, current_retry + 1
            )
            second_half = await self._translate_batch(
                batch[mid:], max_retries, current_retry + 1
            )
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
            await self._load_audio()
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
