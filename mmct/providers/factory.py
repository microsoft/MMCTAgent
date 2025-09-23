from typing import Dict, Type, Any
from loguru import logger

from .base import LLMProvider, EmbeddingProvider, SearchProvider, VisionProvider, TranscriptionProvider
from .azure_providers import AzureLLMProvider, AzureEmbeddingProvider, AzureSearchProvider, AzureVisionProvider, AzureTranscriptionProvider
from .openai_providers import OpenAILLMProvider, OpenAIEmbeddingProvider, OpenAIVisionProvider, OpenAITranscriptionProvider
from ..exceptions import ConfigurationException
from .custom_providers.search_provider import CustomSearchProvider

class ProviderFactory:
    """Factory class for creating provider instances."""
    
    _llm_providers: Dict[str, Type[LLMProvider]] = {
        'azure': AzureLLMProvider,
        'openai': OpenAILLMProvider,
    }
    
    _embedding_providers: Dict[str, Type[EmbeddingProvider]] = {
        'azure': AzureEmbeddingProvider,
        'openai': OpenAIEmbeddingProvider,
    }
    
    _search_providers: Dict[str, Type[SearchProvider]] = {
        'azure_ai_search': AzureSearchProvider,
        'custom_search': CustomSearchProvider
        # Add other search providers here
    }
    
    _vision_providers: Dict[str, Type[VisionProvider]] = {
        'azure': AzureVisionProvider,
        'openai': OpenAIVisionProvider,
    }
    
    _transcription_providers: Dict[str, Type[TranscriptionProvider]] = {
        'azure': AzureTranscriptionProvider,
        'openai': OpenAITranscriptionProvider,
    }
    
    @classmethod
    def create_llm_provider(cls, provider_name: str, config: Dict[str, Any]) -> LLMProvider:
        """
        Create LLM provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            LLMProvider instance
            
        Raises:
            ConfigurationException: If provider is not supported
        """
        if provider_name not in cls._llm_providers:
            raise ConfigurationException(
                f"Unknown LLM provider: {provider_name}. "
                f"Supported providers: {list(cls._llm_providers.keys())}"
            )
        
        provider_class = cls._llm_providers[provider_name]
        logger.info(f"Creating LLM provider: {provider_name}")
        return provider_class(config)
    
    @classmethod
    def create_embedding_provider(cls, provider_name: str, config: Dict[str, Any]) -> EmbeddingProvider:
        """
        Create embedding provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            EmbeddingProvider instance
            
        Raises:
            ConfigurationException: If provider is not supported
        """
        if provider_name not in cls._embedding_providers:
            raise ConfigurationException(
                f"Unknown embedding provider: {provider_name}. "
                f"Supported providers: {list(cls._embedding_providers.keys())}"
            )
        
        provider_class = cls._embedding_providers[provider_name]
        logger.info(f"Creating embedding provider: {provider_name}")
        return provider_class(config)
    
    @classmethod
    def create_search_provider(cls, provider_name: str, config: Dict[str, Any]) -> SearchProvider:
        """
        Create search provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            SearchProvider instance
            
        Raises:
            ConfigurationException: If provider is not supported
        """
        if provider_name not in cls._search_providers:
            raise ConfigurationException(
                f"Unknown search provider: {provider_name}. "
                f"Supported providers: {list(cls._search_providers.keys())}"
            )
        
        provider_class = cls._search_providers[provider_name]
        logger.info(f"Creating search provider: {provider_name}")
        return provider_class(config)
    
    @classmethod
    def create_vision_provider(cls, provider_name: str, config: Dict[str, Any]) -> VisionProvider:
        """
        Create vision provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            VisionProvider instance
            
        Raises:
            ConfigurationException: If provider is not supported
        """
        if provider_name not in cls._vision_providers:
            raise ConfigurationException(
                f"Unknown vision provider: {provider_name}. "
                f"Supported providers: {list(cls._vision_providers.keys())}"
            )
        
        provider_class = cls._vision_providers[provider_name]
        logger.info(f"Creating vision provider: {provider_name}")
        return provider_class(config)
    
    @classmethod
    def create_transcription_provider(cls, provider_name: str, config: Dict[str, Any]) -> TranscriptionProvider:
        """
        Create transcription provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Provider configuration
            
        Returns:
            TranscriptionProvider instance
            
        Raises:
            ConfigurationException: If provider is not supported
        """
        if provider_name not in cls._transcription_providers:
            raise ConfigurationException(
                f"Unknown transcription provider: {provider_name}. "
                f"Supported providers: {list(cls._transcription_providers.keys())}"
            )
        
        provider_class = cls._transcription_providers[provider_name]
        logger.info(f"Creating transcription provider: {provider_name}")
        return provider_class(config)
    
    @classmethod
    def get_supported_providers(cls) -> Dict[str, list]:
        """Get list of supported providers by type."""
        return {
            "llm": list(cls._llm_providers.keys()),
            "embedding": list(cls._embedding_providers.keys()),
            "search": list(cls._search_providers.keys()),
            "vision": list(cls._vision_providers.keys()),
            "transcription": list(cls._transcription_providers.keys())
        }
    
    @classmethod
    def register_llm_provider(cls, name: str, provider_class: Type[LLMProvider]):
        """Register a new LLM provider."""
        cls._llm_providers[name] = provider_class
        logger.info(f"Registered LLM provider: {name}")
    
    @classmethod
    def register_embedding_provider(cls, name: str, provider_class: Type[EmbeddingProvider]):
        """Register a new embedding provider."""
        cls._embedding_providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")
    
    @classmethod
    def register_search_provider(cls, name: str, provider_class: Type[SearchProvider]):
        """Register a new search provider."""
        cls._search_providers[name] = provider_class
        logger.info(f"Registered search provider: {name}")
    
    @classmethod
    def register_vision_provider(cls, name: str, provider_class: Type[VisionProvider]):
        """Register a new vision provider."""
        cls._vision_providers[name] = provider_class
        logger.info(f"Registered vision provider: {name}")
    
    @classmethod
    def register_transcription_provider(cls, name: str, provider_class: Type[TranscriptionProvider]):
        """Register a new transcription provider."""
        cls._transcription_providers[name] = provider_class
        logger.info(f"Registered transcription provider: {name}")


# Global provider factory instance
provider_factory = ProviderFactory()