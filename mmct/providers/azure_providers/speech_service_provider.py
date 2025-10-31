import asyncio
import time
from loguru import logger
from typing import Dict, Any, List
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.providers.base import TranscriptionProvider
from mmct.providers.credentials import AzureCredentials
import azure.cognitiveservices.speech as speechsdk


class AzureSpeechServiceProvider(TranscriptionProvider):
    """Azure Speech Service provider for conversation transcription using Azure Speech SDK."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
        self.speech_config = None  # Will be initialized per-request with token

    def _get_speech_config_with_token(self, language: str = None) -> speechsdk.SpeechConfig:
        """Create Speech SDK configuration with fresh token."""
        try:
            region = self.config.get("region")
            resource_id = self.config.get("resource_id")
            use_managed_identity = self.config.get("use_managed_identity", True)

            if not region:
                raise ConfigurationException("Azure Speech Service region is required")

            if use_managed_identity:
                if not resource_id:
                    raise ConfigurationException("Azure Speech Service resource_id is required for managed identity")

                # Get token for managed identity
                token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
                auth_token = f"aad#{resource_id}#{token.token}"

                speech_config = speechsdk.SpeechConfig(
                    region=region,
                    auth_token=auth_token
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure Speech Service API key is required when managed identity is disabled")

                speech_config = speechsdk.SpeechConfig(
                    region=region,
                    subscription=api_key
                )

            # Set language if provided
            if language:
                speech_config.speech_recognition_language = language

            return speech_config

        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure Speech Service config: {e}")

    async def detect_language(self, audio_path: str, candidate_languages: List[str] = None) -> str:
        """
        Detect language from audio file.

        Args:
            audio_path: Path to audio file
            candidate_languages: List of language codes to detect from (e.g., ["en-IN", "hi-IN"])

        Returns:
            Detected language code
        """
        try:
            if candidate_languages is None:
                candidate_languages = ["en-IN", "hi-IN", "te-IN", "or-IN"]

            speech_config = self._get_speech_config_with_token()
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

            auto_detect_source_language_config = (
                speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=candidate_languages)
            )

            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect_source_language_config,
                audio_config=audio_config,
            )

            result = speech_recognizer.recognize_once()
            auto_detect_source_language_result = speechsdk.AutoDetectSourceLanguageResult(result)
            detected_language = auto_detect_source_language_result.language

            logger.info(f"Detected language: {detected_language}")
            return detected_language

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise ProviderException(f"Language detection failed: {e}")

    async def transcribe_file(self, audio_path: str, language: str = None, phrase_list: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
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
                speech_config=speech_config,
                audio_config=audio_config
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
                    logger.info(f"Transcribed: {rec['text']} [{rec['start_time']}–{rec['end_time']}]")
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
        raise NotImplementedError("Speech SDK only supports file-based transcription. Use transcribe_file() instead.")

    def get_credential(self):
        """Get Azure credentials for token-based auth."""
        return self.credential

    async def close(self):
        """Close the speech service client and cleanup resources."""
        logger.info("Azure Speech Service provider closed")
