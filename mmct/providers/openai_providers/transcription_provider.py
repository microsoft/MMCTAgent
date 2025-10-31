from loguru import logger
from typing import Dict, Any
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.providers.base import TranscriptionProvider
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from openai import AsyncOpenAI, OpenAI


class OpenAITranscriptionProvider(TranscriptionProvider):
    """OpenAI Whisper transcription provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes using OpenAI Whisper."""
        try:
            model = self.config.get("model", "whisper-1")
            
            # Create a temporary file-like object
            import io
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"  # Whisper needs a filename
            
            response = await self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
                **kwargs
            )
            
            return response.text
        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            raise ProviderException(f"OpenAI Whisper transcription failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct audio API access."""
        return self.client

    async def close(self):
        """Close the transcription client and cleanup resources."""
        if self.client:
            logger.info("Closing OpenAI transcription client")
            await self.client.close()
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file using OpenAI Whisper."""
        try:
            model = self.config.get("model", "whisper-1")
            
            with open(audio_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language=language,
                    **kwargs
                )
            
            return response.text
        except Exception as e:
            logger.error(f"OpenAI Whisper file transcription failed: {e}")
            raise ProviderException(f"OpenAI Whisper file transcription failed: {e}")
