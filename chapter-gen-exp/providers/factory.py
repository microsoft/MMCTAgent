from typing import Dict, Type
from loguru import logger

from providers.base import (
    LLMProvider,
    EmbeddingProvider,
    SearchProvider,
    ImageEmbeddingProvider,
)
from providers.azure import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
    AzureSearchProvider,
)
from providers.image_embedding_provider import CustomImageEmbeddingProvider
from settings import MMCTConfig, ImageEmbeddingConfig

class ProviderFactory:
    """Factory class for creating provider instances."""
    
    _llm_providers: Dict[str, Type[LLMProvider]] = {
        'azure': AzureLLMProvider,
    }

    _embedding_providers: Dict[str, Type[EmbeddingProvider]] = {
        'azure': AzureEmbeddingProvider,
    }

    _search_providers: Dict[str, Type[SearchProvider]] = {
        'azure': AzureSearchProvider,
        'azure_ai_search': AzureSearchProvider,
    }

    _image_embedding_providers: Dict[str, Type[ImageEmbeddingProvider]] = {
        'clip': CustomImageEmbeddingProvider,
        'custom_clip': CustomImageEmbeddingProvider,
    }
    
    @classmethod
    def create_llm_provider(cls, provider_name: str = None, enable_cache: bool = True) -> LLMProvider:
        """
        Create LLM provider instance with optional caching.

        Args:
            provider_name: Name of the provider (optional, defaults to config)
            enable_cache: If True, reuse cached instance for better performance (default: True)

        Returns:
            LLMProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.llm.provider

        # Check cache first if caching is enabled
        # if enable_cache and provider_name in cls._llm_provider_cache:
        #     logger.debug(f"Reusing cached LLM provider: {provider_name}")
        #     return cls._llm_provider_cache[provider_name]

        if provider_name not in cls._llm_providers:
            raise RuntimeError(
                f"Unknown LLM provider: {provider_name}. "
                f"Supported providers: {list(cls._llm_providers.keys())}"
            )

        provider_class = cls._llm_providers[provider_name]
        logger.info(f"Creating LLM provider: {provider_name}")
        provider_instance = provider_class(config.llm.model_dump())

        # Cache the instance if caching is enabled
        # if enable_cache:
        #     cls._llm_provider_cache[provider_name] = provider_instance

        return provider_instance
    
    @classmethod
    def create_embedding_provider(cls, provider_name: str = None, enable_cache: bool = True) -> EmbeddingProvider:
        """
        Create embedding provider instance with optional caching.

        Args:
            provider_name: Name of the provider (optional, defaults to config)
            enable_cache: If True, reuse cached instance for better performance (default: True)

        Returns:
            EmbeddingProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.embedding.provider

        # Check cache first if caching is enabled
        # if enable_cache and provider_name in cls._embedding_provider_cache:
        #     logger.debug(f"Reusing cached embedding provider: {provider_name}")
        #     return cls._embedding_provider_cache[provider_name]

        if provider_name not in cls._embedding_providers:
            raise RuntimeError(
                f"Unknown embedding provider: {provider_name}. "
                f"Supported providers: {list(cls._embedding_providers.keys())}"
            )

        provider_class = cls._embedding_providers[provider_name]
        logger.info(f"Creating embedding provider: {provider_name}")
        provider_instance = provider_class(config.embedding.model_dump())

        # Cache the instance if caching is enabled
        # if enable_cache:
        #     cls._embedding_provider_cache[provider_name] = provider_instance

        return provider_instance

    @classmethod
    def create_search_provider(cls, provider_name: str = None) -> SearchProvider:
        """Instantiate configured search provider."""

        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.search.provider

        if provider_name not in cls._search_providers:
            raise RuntimeError(
                f"Unknown search provider: {provider_name}. "
                f"Supported providers: {list(cls._search_providers.keys())}"
            )

        provider_class = cls._search_providers[provider_name]
        logger.info(f"Creating search provider: {provider_name}")
        return provider_class(config.search.model_dump())

    @classmethod
    def create_image_embedding_provider(
        cls,
        provider_name: str = None,
    ) -> ImageEmbeddingProvider:
        """Instantiate an image embedding provider (default: CLIP)."""

        if provider_name is None:
            provider_name = 'clip'

        if provider_name not in cls._image_embedding_providers:
            raise RuntimeError(
                f"Unknown image embedding provider: {provider_name}. "
                f"Supported providers: {list(cls._image_embedding_providers.keys())}"
            )

        provider_class = cls._image_embedding_providers[provider_name]
        logger.info(f"Creating image embedding provider: {provider_name}")
        config = ImageEmbeddingConfig().to_provider_config()
        return provider_class(config)
# Global provider factory instance
provider_factory = ProviderFactory()