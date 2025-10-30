from typing import Dict, Type
from loguru import logger

from .base import (
    LLMProvider,
    EmbeddingProvider,
    SearchProvider,
    VisionProvider,
    TranscriptionProvider,
    StorageProvider
)
from .azure_providers import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
    AzureSearchProvider,
    WhisperTranscriptionProvider,
    AzureSpeechServiceProvider,
    AzureVisionProvider,
    AzureStorageProvider
)
from .openai_providers import (
    OpenAILLMProvider,
    OpenAIEmbeddingProvider,
    OpenAIVisionProvider,
    OpenAITranscriptionProvider
)
from ..utils.error_handler import ConfigurationException
from .custom_providers import (
    CustomSearchProvider
)
from ..config.settings import MMCTConfig

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
        'azure': WhisperTranscriptionProvider,
        'azure_speech': AzureSpeechServiceProvider,
        'openai': OpenAITranscriptionProvider,
    }

    _storage_providers: Dict[str, Type[StorageProvider]] = {
        'azure': AzureStorageProvider,
    }

    @classmethod
    def create_llm_provider(cls, provider_name: str = None) -> LLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider_name: Name of the provider (optional, defaults to config)

        Returns:
            LLMProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.llm.provider

        if provider_name not in cls._llm_providers:
            raise ConfigurationException(
                f"Unknown LLM provider: {provider_name}. "
                f"Supported providers: {list(cls._llm_providers.keys())}"
            )

        provider_class = cls._llm_providers[provider_name]
        logger.info(f"Creating LLM provider: {provider_name}")
        return provider_class(config.llm.model_dump())
    
    @classmethod
    def create_embedding_provider(cls, provider_name: str = None) -> EmbeddingProvider:
        """
        Create embedding provider instance.

        Args:
            provider_name: Name of the provider (optional, defaults to config)

        Returns:
            EmbeddingProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.embedding.provider

        if provider_name not in cls._embedding_providers:
            raise ConfigurationException(
                f"Unknown embedding provider: {provider_name}. "
                f"Supported providers: {list(cls._embedding_providers.keys())}"
            )

        provider_class = cls._embedding_providers[provider_name]
        logger.info(f"Creating embedding provider: {provider_name}")
        return provider_class(config.embedding.model_dump())
    
    @classmethod
    def create_search_provider(cls, provider_name: str = None) -> SearchProvider:
        """
        Create search provider instance.

        Args:
            provider_name: Name of the provider (optional, defaults to config)

        Returns:
            SearchProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.search.provider

        if provider_name not in cls._search_providers:
            raise ConfigurationException(
                f"Unknown search provider: {provider_name}. "
                f"Supported providers: {list(cls._search_providers.keys())}"
            )

        provider_class = cls._search_providers[provider_name]
        logger.info(f"Creating search provider: {provider_name}")
        return provider_class(config.search.model_dump())
    
    @classmethod
    def create_vision_provider(cls, provider_name: str = None) -> VisionProvider:
        """
        Create vision provider instance.

        Args:
            provider_name: Name of the provider (optional, defaults to config)

        Returns:
            VisionProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.vision.provider

        if provider_name not in cls._vision_providers:
            raise ConfigurationException(
                f"Unknown vision provider: {provider_name}. "
                f"Supported providers: {list(cls._vision_providers.keys())}"
            )

        provider_class = cls._vision_providers[provider_name]
        logger.info(f"Creating vision provider: {provider_name}")
        return provider_class(config.vision.model_dump())
    
    @classmethod
    def create_transcription_provider(cls, provider_name: str = None) -> TranscriptionProvider:
        """
        Create transcription provider instance.

        Args:
            provider_name: Name of the provider (optional, defaults to config)

        Returns:
            TranscriptionProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.transcription.provider

        if provider_name not in cls._transcription_providers:
            raise ConfigurationException(
                f"Unknown transcription provider: {provider_name}. "
                f"Supported providers: {list(cls._transcription_providers.keys())}"
            )

        provider_class = cls._transcription_providers[provider_name]
        logger.info(f"Creating transcription provider: {provider_name}")
        return provider_class(config.transcription.model_dump())

    @classmethod
    def create_storage_provider(cls, provider_name: str = None) -> StorageProvider:
        """
        Create storage provider instance.

        Args:
            provider_name: Name of the provider (optional, defaults to config)

        Returns:
            StorageProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.storage.provider

        if provider_name not in cls._storage_providers:
            raise ConfigurationException(
                f"Unknown storage provider: {provider_name}. "
                f"Supported providers: {list(cls._storage_providers.keys())}"
            )

        provider_class = cls._storage_providers[provider_name]
        logger.info(f"Creating storage provider: {provider_name}")
        return provider_class(config.storage.model_dump())

    @classmethod
    def get_supported_providers(cls) -> Dict[str, list]:
        """Get list of supported providers by type."""
        return {
            "llm": list(cls._llm_providers.keys()),
            "embedding": list(cls._embedding_providers.keys()),
            "search": list(cls._search_providers.keys()),
            "vision": list(cls._vision_providers.keys()),
            "transcription": list(cls._transcription_providers.keys()),
            "storage": list(cls._storage_providers.keys())
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

    @classmethod
    def register_storage_provider(cls, name: str, provider_class: Type[StorageProvider]):
        """Register a new storage provider."""
        cls._storage_providers[name] = provider_class
        logger.info(f"Registered storage provider: {name}")


# Global provider factory instance
provider_factory = ProviderFactory()