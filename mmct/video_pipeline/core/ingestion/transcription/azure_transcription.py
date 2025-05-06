import asyncio
import time
import os
import aiofiles
import json
from typing import List, Optional, Dict, Any
from loguru import logger
from mmct.video_pipeline.core.ingestion.models import TranslationResponse
import azure.cognitiveservices.speech as speechsdk
from azure.identity.aio import DefaultAzureCredential
from mmct.video_pipeline.core.ingestion.transcription.base_transcription import (
    Transcription,
)
from mmct.video_pipeline.utils.helper import extract_wav_from_video
from mmct.video_pipeline.core.ingestion.languages import Languages
from mmct.video_pipeline.utils.helper import get_media_folder
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

class AzureTranscription(Transcription):
    def __init__(self, video_path: str, hash_id: str, language:str=None) -> None:
        super().__init__(video_path=video_path, hash_id=hash_id, language=language)
        self.audio_container = os.getenv("AUDIO_CONTAINER")
        self.local_save = []

    async def _load_audio(self):
        try:
            logger.info("Extracting the audio from the video")
            self.audio_path = os.path.join(await get_media_folder(), f"{self.hash_id}.wav")

            await extract_wav_from_video(
                video_path=self.video_path, output_path=self.audio_path
            )
            self.local_save.append(self.audio_path)
        except Exception as e:
            logger.exception(f"Error loading audio, {e}")
            raise 

    async def detect_language(self):
        try:
            token = await DefaultAzureCredential().get_token(
                "https://cognitiveservices.azure.com/.default"
            )
            token = token.token
            resource_id = os.getenv('AZURE_SPEECH_SERVICE_RESOURCE_ID')
            token = "aad#" + resource_id + "#" + token
            speech_config = speechsdk.SpeechConfig(region=os.getenv('AZURE_SPEECH_SERVICE_REGION'), auth_token=token)
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

    async def get_transcript(self):
        try:
            result = []
            async with DefaultAzureCredential() as credential:
                token = await credential.get_token(
                    "https://cognitiveservices.azure.com/.default"
                )
            token = token.token
            resource_id = os.getenv('AZURE_SPEECH_SERVICE_RESOURCE_ID')
            auth_token = f"aad#{resource_id}#{token}"
            speech_config = speechsdk.SpeechConfig(
                region=os.getenv('AZURE_SPEECH_SERVICE_REGION'), auth_token=auth_token
            )
            logger.info("Speech Config initialized")
            if self.source_language == None:
                lang = await self.detect_language()
                self.source_language['lang-code'] = lang if lang not in [None, "Unknown"] else "en-IN"
                self.source_language['lang'] = Languages(self.source_language['lang-code']).name
            speech_config.speech_recognition_language = self.source_language['lang-code']
            audio_config = speechsdk.audio.AudioConfig(filename=self.audio_path)

            transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config, audio_config=audio_config
            )

            # 3) (Optional) Add your phrase list -- Only available for Hindi right now
            if self.source_language['lang-code'] == "hi-IN":
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

    async def _get_translated_transcript(self, text: str,  max_chars_per_batch: int = 2000) -> TranslationResponse:
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
                
                
            prompt = """You are brilliant transcript translator, your task is to translate the given transcript from {source_language} to English.

            # Note: 
            - Do not remove any part of original transcript, just translate the segment of the transcript.
            - given transcript will contain the text with their timestamps.
            - given transcripts can be in different dialects of {source_language}, so be careful while translating.
            - Retain the count and the timestamp that are there in the given transcript.
            """
            prompt = prompt.format(source_language=self.source_language['lang'].split("_")[0].capitalize())
            logger.info("Inserting the source language to prompt")
            if self.source_language['lang-code'] == "hi-IN":
                logger.info("Adding glossary for hindi vocabulary")
                glossary_table = self.glossary_df[
                    self.glossary_df["hindi_terms"].apply(lambda term: term in text)
                ].to_markdown(index=False)
                prompt += f"\n\n# Glossary:\n{glossary_table}\n"
                
            all_translations: List[str] = []
            logger.info("Translating the texts batchwise")
            for batch in batches:
                to_translate = json.dumps([e["text"] for e in batch], ensure_ascii=False)
                messages=[
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
                        "AZURE_OPENAI_MODEL"
                        if os.getenv("LLM_PROVIDER") == "azure"
                        else "OPENAI_MODEL"
                    ),
                    messages=messages,
                    temperature=0,
                    top_p=0.1,
                    response_format=TranslationResponse
                )
                
                translation_response: TranslationResponse = response.choices[0].message.parsed
                all_translations.extend(translation_response.translations)
            
            logger.info("Succesfully translated the batchwise text chunks")    
            # Reassemble into SRT format
            output_blocks = []
            for entry, translation in zip(entries, all_translations):
                block = "\n".join([entry["seq"], entry["time"], translation.strip()])
                output_blocks.append(block)
            logger.info("Reassembled the batchwise translated chunks to SRT format")
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
            if self.source_language['lang-code'] != "en-IN":
                transcript = await self._get_translated_transcript(
                    transcript
                )  # translating to english
                logger.info(f"translated {transcript}")
            transcript_save_path = os.path.join(await get_media_folder(),f"transcript_{self.hash_id}.srt")
            async with aiofiles.open(
                transcript_save_path, "w", encoding="utf-8"
            ) as f:
                await f.write(transcript)
                
            self.local_save.append(transcript_save_path)

            return transcript, self.local_save
        except Exception as e:
            logger.exception(f"Exception occured : {e}")
            raise 

if __name__ == "__main__":
    video_path = "C:/Users/v-amanpatkar/Downloads/sample_video2.mp4"
    hash_id = "abcd"
    transcriber = AzureTranscription(video_path=video_path, hash_id=hash_id)
    transcript = asyncio.run(transcriber.run())
    print("Final Transcript:")
    print(transcript)
