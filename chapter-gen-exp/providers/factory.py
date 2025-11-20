from typing import Dict, Type
from loguru import logger

from providers.base import (
    LLMProvider,
    EmbeddingProvider,
)
from providers.azure import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
)
from settings import MMCTConfig

class ProviderFactory:
    """Factory class for creating provider instances."""
    
    _llm_providers: Dict[str, Type[LLMProvider]] = {
        'azure': AzureLLMProvider,
    }
    
    _embedding_providers: Dict[str, Type[EmbeddingProvider]] = {
        'azure': AzureEmbeddingProvider,
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
# Global provider factory instance
provider_factory = ProviderFactory()