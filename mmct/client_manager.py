from typing import Dict, Any, Optional, List
from loguru import logger

from .config.settings import MMCTConfig
from .providers.factory import ProviderFactory
from .providers.base import LLMProvider, SearchProvider, VisionProvider, TranscriptionProvider
from .security.secrets import SecretsManager
from .utils.logging_config import LoggingConfig, get_logger
from .exceptions import ConfigurationException


class ClientManager:
    """
    Unified client manager for all MMCT providers.
    
    This class replaces the old LLMClient and provides a modern, 
    vendor-agnostic interface for all AI services.
    """
    
    def __init__(self, config: Optional[MMCTConfig] = None):
        self.config = config or MMCTConfig()
        self.logger = get_logger("ClientManager")
        self.secrets_manager = None
        self.providers = {}
        
        # Initialize logging
        LoggingConfig.setup_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.log_file,
            enable_json=self.config.logging.enable_json,
            enable_file_logging=self.config.logging.enable_file_logging,
            max_file_size=self.config.logging.max_file_size,
            retention_days=self.config.logging.retention_days,
            app_name=self.config.app_name
        )
        
        # Initialize secrets manager if enabled
        if self.config.security.enable_secrets_manager:
            self.secrets_manager = SecretsManager(
                vault_url=self.config.security.keyvault_url,
                use_managed_identity=True
            )
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all configured providers."""
        try:
            # Initialize LLM provider
            self.providers['llm'] = self._create_llm_provider()
            
            # Initialize search provider
            self.providers['search'] = self._create_search_provider()
            
            # Initialize vision provider
            self.providers['vision'] = self._create_vision_provider()
            
            # Initialize transcription provider
            self.providers['transcription'] = self._create_transcription_provider()
            
            self.logger.info("All providers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize providers: {e}")
            raise ConfigurationException(f"Provider initialization failed: {e}")
    
    def _create_llm_provider(self) -> LLMProvider:
        """Create LLM provider based on configuration."""
        provider_name = self.config.llm.provider
        
        # Build provider configuration
        provider_config = {
            "endpoint": self.config.llm.endpoint,
            "deployment_name": self.config.llm.deployment_name,
            "api_version": self.config.llm.api_version,
            "model_name": self.config.llm.model_name,
            "use_managed_identity": self.config.llm.use_managed_identity,
            "timeout": self.config.llm.timeout,
            "max_retries": self.config.llm.max_retries,
            # "temperature": self.config.llm.temperature,
        }
        
        # Handle API key
        if not self.config.llm.use_managed_identity:
            api_key = self.config.llm.api_key
            if self.secrets_manager:
                api_key = self.secrets_manager.get_secret("LLM_API_KEY", api_key)
            provider_config["api_key"] = api_key
        
        return ProviderFactory.create_llm_provider(provider_name, provider_config)
    
    def _create_search_provider(self) -> SearchProvider:
        """Create search provider based on configuration."""
        provider_name = self.config.search.provider
        
        # Build provider configuration
        provider_config = {
            "endpoint": self.config.search.endpoint,
            "index_name": self.config.search.index_name,
            "use_managed_identity": self.config.search.use_managed_identity,
            "timeout": self.config.search.timeout,
        }
        
        # Handle API key
        if not self.config.search.use_managed_identity:
            api_key = self.config.search.api_key
            if self.secrets_manager:
                api_key = self.secrets_manager.get_secret("SEARCH_API_KEY", api_key)
            provider_config["api_key"] = api_key
        
        return ProviderFactory.create_search_provider(provider_name, provider_config)
    
    def _create_vision_provider(self) -> VisionProvider:
        """Create vision provider based on configuration."""
        # For now, use LLM provider for vision
        provider_name = self.config.llm.provider
        
        # Build provider configuration
        provider_config = {
            "endpoint": self.config.llm.endpoint,
            "deployment_name": self.config.llm.vision_deployment_name or self.config.llm.deployment_name,
            "api_version": self.config.llm.vision_api_version or self.config.llm.api_version,
            "model_name": self.config.llm.model_name,
            "use_managed_identity": self.config.llm.use_managed_identity,
            "timeout": self.config.llm.timeout,
            "max_retries": self.config.llm.max_retries,
        }
        
        # Handle API key
        if not self.config.llm.use_managed_identity:
            api_key = self.config.llm.api_key
            if self.secrets_manager:
                api_key = self.secrets_manager.get_secret("LLM_API_KEY", api_key)
            provider_config["api_key"] = api_key
        
        return ProviderFactory.create_vision_provider(provider_name, provider_config)
    
    def _create_transcription_provider(self) -> TranscriptionProvider:
        """Create transcription provider based on configuration."""
        provider_name = self.config.transcription.provider
        
        # Build provider configuration
        provider_config = {
            "endpoint": self.config.transcription.endpoint,
            "deployment_name": self.config.transcription.deployment_name,
            "api_version": self.config.transcription.api_version,
            "use_managed_identity": self.config.transcription.use_managed_identity,
            "timeout": self.config.transcription.timeout,
        }
        
        # Handle API key
        if not self.config.transcription.use_managed_identity:
            api_key = self.config.transcription.api_key
            if self.secrets_manager:
                api_key = self.secrets_manager.get_secret("SPEECH_SERVICE_KEY", api_key)
            provider_config["api_key"] = api_key
        
        return ProviderFactory.create_transcription_provider(provider_name, provider_config)
    
    # Convenience methods for accessing providers
    
    def get_llm_provider(self) -> LLMProvider:
        """Get the LLM provider."""
        return self.providers['llm']
    
    def get_search_provider(self) -> SearchProvider:
        """Get the search provider."""
        return self.providers['search']
    
    def get_vision_provider(self) -> VisionProvider:
        """Get the vision provider."""
        return self.providers['vision']
    
    def get_transcription_provider(self) -> TranscriptionProvider:
        """Get the transcription provider."""
        return self.providers['transcription']
    
    # High-level convenience methods
    
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using the configured LLM provider."""
        return await self.get_llm_provider().chat_completion(messages, **kwargs)
    
    async def generate_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using the configured LLM provider."""
        return await self.get_llm_provider().embedding(text, **kwargs)
    
    async def search_documents(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """Search documents using the configured search provider."""
        return await self.get_search_provider().search(query, index_name, **kwargs)
    
    async def analyze_image(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """Analyze image using the configured vision provider."""
        return await self.get_vision_provider().analyze_image(image_data, **kwargs)
    
    async def transcribe_audio(self, audio_path: str, language: str = None, **kwargs) -> str:
        """Transcribe audio using the configured transcription provider."""
        return await self.get_transcription_provider().transcribe_file(audio_path, language, **kwargs)
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about currently configured providers."""
        return {
            "llm": self.config.llm.provider,
            "search": self.config.search.provider,
            "transcription": self.config.transcription.provider,
            "environment": self.config.environment,
            "app_version": self.config.app_version
        }


# Global client manager instance
_client_manager = None


def get_client_manager() -> ClientManager:
    """Get the global client manager instance."""
    global _client_manager
    if _client_manager is None:
        _client_manager = ClientManager()
    return _client_manager


def initialize_client_manager(config: Optional[MMCTConfig] = None) -> ClientManager:
    """Initialize the global client manager with custom configuration."""
    global _client_manager
    _client_manager = ClientManager(config)
    return _client_manager