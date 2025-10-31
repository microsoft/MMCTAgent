from loguru import logger
from typing import Dict, Any
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.providers.base import TranscriptionProvider
from mmct.providers.credentials import AzureCredentials
from openai import AsyncAzureOpenAI
from azure.identity import get_bearer_token_provider

class WhisperTranscriptionProvider(TranscriptionProvider):
    """Azure OpenAI Whisper transcription provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_credentials()
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize Azure OpenAI client for Whisper."""
        try:
            logger.info(f"Transcription config received: {self.config}")
            endpoint = self.config.get("endpoint")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)

            logger.info(f"Endpoint from config: {endpoint}")
            if not endpoint:
                raise ConfigurationException("Azure OpenAI endpoint is required for Whisper API")

            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential,
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    timeout=timeout
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")

                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    timeout=timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client for transcription: {e}")

    async def transcribe(self, audio_data: bytes, language: str = None, **kwargs) -> str:
        """Transcribe audio bytes to text using Whisper."""
        raise NotImplementedError("Whisper API requires file-based transcription. Use transcribe_file() instead.")

    async def transcribe_file(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio file to text using Azure OpenAI Whisper."""
        try:
            deployment_name = self.config.get("deployment_name")
            if not deployment_name:
                raise ConfigurationException("Azure OpenAI Whisper deployment name is required")

            response_format = kwargs.get("response_format", "text")

            with open(audio_path, "rb") as audio_file:
                result = await self.client.audio.translations.create(
                    file=audio_file,
                    model=deployment_name,
                    response_format=response_format
                )

            return result
        except Exception as e:
            logger.error(f"Azure Whisper transcription failed: {e}")
            raise ProviderException(f"Azure Whisper transcription failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct audio API access."""
        return self.client

    async def close(self):
        """Close the transcription client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI transcription client")
            await self.client.close()
