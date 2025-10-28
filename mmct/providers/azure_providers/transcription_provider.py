from loguru import logger
from typing import Dict, Any
from mmct.utils.error_handler import ProviderException, ConfigurationException
from mmct.providers.base import TranscriptionProvider
from azure.cognitiveservices.speech import SpeechConfig
from mmct.providers.credentials import AzureCredentials

class AzureTranscriptionProvider(TranscriptionProvider):
    """Azure Speech Service transcription provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_async_credentials()
        self.speech_config = self._initialize_speech_config()
    
    def _initialize_speech_config(self):
        """Initialize Azure Speech configuration."""
        try:
            endpoint = self.config.get("endpoint")
            use_managed_identity = self.config.get("use_managed_identity", True)
            
            if not endpoint:
                raise ConfigurationException("Azure Speech Service endpoint is required")
            
            if use_managed_identity:
                # For managed identity, we need to use token authentication
                # This is a simplified implementation
                return SpeechConfig(
                    endpoint=endpoint,
                    auth_token=self._get_auth_token()
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure Speech Service API key is required when managed identity is disabled")
                
                return SpeechConfig(
                    endpoint=endpoint,
                    subscription=api_key
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure Speech config: {e}")