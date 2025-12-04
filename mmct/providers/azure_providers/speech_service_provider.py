import asyncio
import time
import os
import json
import aiofiles
from loguru import logger
from typing import Dict, Any, List, Union, Optional, Tuple
from mmct.utils.error_handler import ProviderException, ConfigurationException, handle_exceptions, convert_exceptions
from mmct.providers.base import BaseTranscriptionProvider, BaseLLMProvider
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
import azure.cognitiveservices.speech as speechsdk


class AzureSpeechServiceProvider(BaseTranscriptionProvider):
    """Azure Speech Service provider for conversation transcription using Azure Speech SDK."""

    def __init__(
        self,
        speech_service_region: str,
        speech_service_resource_id: str,
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        speech_config: Optional[dict] = None,
        api_key: Optional[str] = None,
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        if not speech_service_region:
            raise ConfigurationException("Speech service region is required!")

        if not speech_service_resource_id:
            raise ConfigurationException("Speech service resource Id is required!")

        # Validate that exactly one of credentials or api_key is provided
        if credentials is None and api_key is None:
            raise ConfigurationException("Either credentials or api_key must be provided!")

        if credentials is not None and api_key is not None:
            raise ConfigurationException("Only one of credentials or api_key should be provided, not both!")

        self.credential = credentials
        self.speech_config = speech_config  # Will be initialized per-request with token
        self.speech_service_region = speech_service_region
        self.speech_service_resource_id = speech_service_resource_id
        self.api_key = api_key
        self.llm_provider = llm_provider  # Optional LLM provider for translation

    def _get_speech_config_with_token(self, language: str = None) -> speechsdk.SpeechConfig:
        """Create Speech SDK configuration with fresh token."""
        try:
            if self.credentials is not None:
                # Get token for managed identity
                token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
                auth_token = f"aad#{self.speech_service_resource_id}#{token.token}"
                speech_config = speechsdk.SpeechConfig(region=self.speech_service_region, auth_token=auth_token)
            else:
                speech_config = speechsdk.SpeechConfig(region=self.speech_service_region, subscription=self.api_key)

            # Set language if provided
            if language:
                speech_config.speech_recognition_language = language

            return speech_config

        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure Speech Service config: {e}")

    async def transcribe_file(
        self, audio_path: str, language: str = None, phrase_list: List[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio file using Azure Speech Service with conversation transcription.

        Args:
            audio_path: Path to audio file (.wav format)
            language: Language code (e.g., "en-IN", "hi-IN")
            phrase_list: Optional list of phrases to boost recognition accuracy
            **kwargs: Additional arguments

        Returns:
            List of transcription segments with text, start_time, end_time, speaker_id
        """
        try:
            result = []

            # Get speech config with token
            speech_config = self._get_speech_config_with_token(language=language)

            logger.info(f"Speech Config initialized with language: {language}")

            # Setup audio config
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

            # Create conversation transcriber
            transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config, audio_config=audio_config
            )

            # Add phrase list if provided (for better recognition)
            if phrase_list:
                phrase_grammar = speechsdk.PhraseListGrammar.from_recognizer(transcriber)
                for phrase in phrase_list[:500]:  # Limit to 500 phrases
                    phrase_grammar.addPhrase(phrase)
                logger.info(f"Added {min(len(phrase_list), 500)} phrases to grammar")

            # Prepare asyncio event for completion
            loop = asyncio.get_running_loop()
            done_evt = asyncio.Event()

            def _stop_cb(evt: speechsdk.SessionEventArgs):
                loop.call_soon_threadsafe(done_evt.set)

            # Define callbacks
            def _on_session_started(evt):
                logger.info("Transcription session started")

            def _on_session_stopped(evt):
                logger.info("Transcription session stopped")
                _stop_cb(evt)

            def _on_canceled(evt):
                logger.info("Transcription canceled")
                _stop_cb(evt)

            def _on_transcribed(evt: speechsdk.SpeechRecognitionEventArgs):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    start_ms = evt.result.offset
                    dur_ms = evt.result.duration
                    # Convert from 100-nanos to seconds
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
                    logger.warning(f"NoMatch: {evt.result.no_match_details}")

            # Connect callbacks
            transcriber.session_started.connect(_on_session_started)
            transcriber.session_stopped.connect(_on_session_stopped)
            transcriber.canceled.connect(_on_canceled)
            transcriber.transcribed.connect(_on_transcribed)

            # Start transcription
            start_future = transcriber.start_transcribing_async()
            start_future.get()  # Wait for SDK to begin

            # Wait until SDK signals stop/cancel
            await done_evt.wait()

            # Clean shutdown
            stop_future = transcriber.stop_transcribing_async()
            stop_future.get()

            logger.info(f"Transcription completed with {len(result)} segments")

            if not result:
                logger.warning("No transcription results obtained!")

            return result

        except Exception as e:
            logger.error(f"Azure Speech transcription failed: {e}")
            raise ProviderException(f"Azure Speech transcription failed: {e}")

    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes to text (not implemented for Speech SDK)."""
        raise NotImplementedError(
            "Speech SDK only supports file-based transcription. Use transcribe_file() instead."
        )

    async def _extract_audio_from_video(self, video_path: str, output_path: str) -> None:
        """
        Extract WAV audio from video file using FFmpeg.

        Args:
            video_path: Path to the video file
            output_path: Path where audio file should be saved (.wav format)

        Raises:
            ProviderException: If audio extraction fails
        """
        try:
            logger.info(f"Extracting WAV audio from video: {video_path}")

            # Verify video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Run FFmpeg in a subprocess to extract WAV audio
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i",
                video_path,
                "-acodec",
                "pcm_s16le",  # WAV format
                "-ar",
                "16000",  # Sample rate 16kHz (good for speech recognition)
                "-ac",
                "1",  # Mono
                output_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Wait for the process to complete
            returncode = await process.wait()

            if returncode != 0:
                raise ProviderException(f"FFmpeg failed with return code {returncode}")

            # Verify audio file was created
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Audio extraction failed - output file not created: {output_path}")

            audio_size = os.path.getsize(output_path)
            logger.info(f"Successfully extracted WAV audio to: {output_path} (size: {audio_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to extract audio from video: {e}")
            raise ProviderException(f"Failed to extract audio from video: {e}")

    async def _format_transcript(self, transcript_segments: List[Dict[str, Any]]) -> str:
        """
        Format transcription segments into SRT format.

        Args:
            transcript_segments: List of transcription segments from transcribe_file()

        Returns:
            Formatted SRT transcript string
        """
        try:
            logger.info("Formatting transcript segments to SRT format")

            if not isinstance(transcript_segments, list):
                raise ValueError("transcript_segments must be a list")

            formatted_transcript = ""
            for idx, segment in enumerate(transcript_segments, 1):
                if not segment.get("text"):
                    continue

                formatted_transcript += (
                    f"{idx}\n"
                    f"{segment['start_time']},000 --> {segment['end_time']},000\n"
                    f"{segment['text']}\n\n"
                )

            logger.info(f"Successfully formatted {len(transcript_segments)} segments")
            return formatted_transcript

        except Exception as e:
            logger.error(f"Transcript formatting failed: {e}")
            raise ProviderException(f"Transcript formatting failed: {e}")

    async def _translate_batch(
        self,
        batch: List[Dict[str, str]],
        source_language: str,
        max_retries: int = 3,
        current_retry: int = 0
    ) -> List[str]:
        """
        Translate a batch of text using LLM provider with retry logic.

        Args:
            batch: List of dict entries with 'text' key
            source_language: Source language name (e.g., "Hindi", "Telugu")
            max_retries: Maximum number of retries
            current_retry: Current retry count

        Returns:
            List of translated text strings
        """
        if not self.llm_provider:
            raise ProviderException("LLM provider is required for translation but not provided")

        try:
            # Import here to avoid circular dependency
            from mmct.video_pipeline.core.ingestion.models import TranslationResponse

            to_translate = json.dumps([e["text"] for e in batch], ensure_ascii=False)
            logger.info(f"Translating batch of {len(batch)} entries (retry {current_retry}/{max_retries})")

            prompt = f"""You are a highly skilled translator. Your task is to translate the provided JSON array of text from {source_language} to English with utmost accuracy.

# Instructions:
- Translate each line of the input text exactly as it is, without adding, omitting, or altering any information.
- The input text may include different dialects of {source_language}; translate them carefully while preserving the original meaning.
- Do not hallucinate or introduce any new information that is not present in the input text.
- If a term or phrase is unclear, translate it as closely as possible to its original meaning without making assumptions.
"""

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": prompt}],
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
                first_half = await self._translate_batch(
                    batch[:mid], source_language, max_retries, current_retry + 1
                )
                second_half = await self._translate_batch(
                    batch[mid:], source_language, max_retries, current_retry + 1
                )
                return first_half + second_half

            return translations

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise ProviderException(f"Batch translation failed: {e}")

    async def _translate_transcript(
        self, srt_text: str, source_language: str, max_chars_per_batch: int = 2000
    ) -> str:
        """
        Translate an SRT formatted transcript to English.

        Args:
            srt_text: SRT formatted transcript text
            source_language: Source language name (e.g., "Hindi", "Telugu")
            max_chars_per_batch: Maximum characters per translation batch

        Returns:
            Translated SRT transcript
        """
        if not self.llm_provider:
            logger.warning("LLM provider not provided, skipping translation")
            return srt_text

        try:
            logger.info(f"Translating transcript from {source_language} to English")

            # Parse SRT blocks
            raw_blocks = [b.strip() for b in srt_text.strip().split("\n\n") if b.strip()]
            entries = []

            for block in raw_blocks:
                lines = block.splitlines()
                if len(lines) < 3:
                    continue
                seq_no = lines[0]
                timestamp = lines[1]
                content = "\n".join(lines[2:])
                entries.append({"seq": seq_no, "time": timestamp, "text": content})

            # Batch entries by text length
            batches: List[List[Dict]] = []
            curr_batch, curr_len = [], 0

            for entry in entries:
                length = len(entry["text"])
                if curr_len + length > max_chars_per_batch and curr_batch:
                    batches.append(curr_batch)
                    curr_batch, curr_len = [], 0
                curr_batch.append(entry)
                curr_len += length

            if curr_batch:
                batches.append(curr_batch)

            logger.info(f"Split transcript into {len(batches)} batches for translation")

            # Translate each batch
            all_translations: List[str] = []
            for batch in batches:
                batch_translations = await self._translate_batch(batch, source_language)
                all_translations.extend(batch_translations)

            # Reassemble into SRT format
            output_blocks = []
            for entry, translation in zip(entries, all_translations):
                block = "\n".join([entry["seq"], entry["time"], translation.strip()])
                output_blocks.append(block)

            translated_srt = "\n\n".join(output_blocks)
            logger.info("Successfully translated transcript")
            return translated_srt

        except Exception as e:
            logger.error(f"Transcript translation failed: {e}")
            raise ProviderException(f"Transcript translation failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_video(
        self,
        video_path: str,
        hash_id: str,
        output_dir: Optional[str] = None,
        language: str = None,
        **kwargs
    ) -> Tuple[str, List[str]]:
        """
        Transcribe video file by extracting audio and transcribing it using Azure Speech Service.
        Optionally translates to English if source language is not English.

        Args:
            video_path: Path to the video file
            hash_id: Unique identifier for the video
            output_dir: Directory to save audio and transcript files (defaults to current working directory's media folder)
            language: Language code for transcription (e.g., "en-IN", "hi-IN"). **Required parameter**.
            **kwargs: Additional parameters:
                - phrase_list: List of phrases to boost recognition accuracy
                - translate_to_english: Whether to translate non-English transcripts to English (default: True)
                - source_language_name: Human-readable source language name for translation (e.g., "Hindi")

        Returns:
            Tuple of (transcript_content, list_of_local_file_paths)
            - transcript_content: The transcribed (and optionally translated) text in SRT format
            - list_of_local_file_paths: List of paths to temporary files created (audio, transcript)

        Raises:
            ProviderException: If transcription fails or language parameter is not provided
        """
        local_files = []

        try:
            # Determine output directory
            if output_dir is None:
                output_dir = os.path.join(os.getcwd(), "media")
                os.makedirs(output_dir, exist_ok=True)

            # Validate language parameter
            if not language:
                raise ProviderException(
                    "Language parameter is required for Azure Speech Service transcription. "
                    "Please provide a valid language code (e.g., 'en-IN', 'hi-IN', 'te-IN')."
                )

            # Step 1: Extract WAV audio from video
            audio_path = os.path.join(output_dir, f"{hash_id}.wav")
            await self._extract_audio_from_video(video_path, audio_path)
            local_files.append(audio_path)
            logger.info(f"Audio extracted and saved to: {audio_path}")

            # Step 2: Transcribe the audio file
            phrase_list = kwargs.get("phrase_list")
            logger.info(f"Starting transcription with language: {language}")

            transcript_segments = await self.transcribe_file(
                audio_path=audio_path,
                language=language,
                phrase_list=phrase_list
            )
            logger.info(f"Successfully generated {len(transcript_segments)} transcript segments")

            # Step 3: Format transcript to SRT
            transcript_srt = await self._format_transcript(transcript_segments)
            logger.info("Successfully formatted transcript to SRT")

            # Step 4: Translate if needed and LLM provider is available
            translate_to_english = kwargs.get("translate_to_english", True)
            if translate_to_english and language != "en-IN" and self.llm_provider:
                # Extract language name from code (e.g., "hi-IN" -> "Hindi")
                source_language_name = kwargs.get("source_language_name")
                if not source_language_name:
                    # Try to infer from language code
                    lang_map = {
                        "hi-IN": "Hindi",
                        "te-IN": "Telugu",
                        "or-IN": "Odia",
                        "ta-IN": "Tamil",
                        "bn-IN": "Bengali"
                    }
                    source_language_name = lang_map.get(language, "the source language")

                logger.info(f"Translating transcript from {source_language_name} to English")
                transcript_srt = await self._translate_transcript(transcript_srt, source_language_name)
                logger.info("Successfully translated transcript to English")

            # Step 6: Save transcript to file
            transcript_path = os.path.join(output_dir, f"transcript_{hash_id}.srt")
            async with aiofiles.open(transcript_path, "w", encoding="utf-8") as f:
                await f.write(transcript_srt)
            local_files.append(transcript_path)
            logger.info(f"Transcript saved to: {transcript_path}")

            return transcript_srt, local_files

        except Exception as e:
            # Clean up any files created before the error
            for file_path in local_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up file after error: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file {file_path}: {cleanup_error}")

            logger.error(f"Video transcription failed: {e}")
            raise ProviderException(f"Video transcription failed: {e}")

    async def close(self):
        """Close the speech service client and cleanup resources."""
        try:
            # Close credential if it has a close method
            if self.credential is not None and hasattr(self.credential, 'close'):
                if asyncio.iscoroutinefunction(self.credential.close):
                    await self.credential.close()
                else:
                    self.credential.close()
                self.credential = None

            logger.info("Azure Speech Service provider closed")
        except Exception as e:
            logger.error(f"Error closing Azure Speech Service provider: {e}")
            raise