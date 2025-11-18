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
    CustomSearchProvider,
    LocalFaissSearchProvider,
    LocalStorageProvider
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
        'custom_search': CustomSearchProvider,
        'local_faiss': LocalFaissSearchProvider
        # Add other search providers here
    }
    
    # Cache for reusable provider instances (singleton pattern)
    # _llm_provider_cache: Dict[str, LLMProvider] = {}
    # _search_provider_cache: Dict[str, SearchProvider] = {}
    # _embedding_provider_cache: Dict[str, EmbeddingProvider] = {}
    
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
        'local': LocalStorageProvider
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
            raise ConfigurationException(
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
            raise ConfigurationException(
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
    def create_search_provider(cls, provider_name: str = None, enable_cache: bool = True) -> SearchProvider:
        """
        Create search provider instance with optional caching.

        Args:
            provider_name: Name of the provider (optional, defaults to config)
            enable_cache: If True, reuse cached instance for better performance (default: True)

        Returns:
            SearchProvider instance

        Raises:
            ConfigurationException: If provider is not supported
        """
        config = MMCTConfig()
        if provider_name is None:
            provider_name = config.search.provider

        # Check cache first if caching is enabled
        # if enable_cache and provider_name in cls._search_provider_cache:
        #     logger.debug(f"Reusing cached search provider: {provider_name}")
        #     return cls._search_provider_cache[provider_name]

        if provider_name not in cls._search_providers:
            raise ConfigurationException(
                f"Unknown search provider: {provider_name}. "
                f"Supported providers: {list(cls._search_providers.keys())}"
            )

        provider_class = cls._search_providers[provider_name]
        logger.info(f"Creating search provider: {provider_name}")
        # instantiate provider with configured dict
        provider_instance = provider_class(config.search.model_dump())
        # tag provider instance with an explicit provider name for robust detection
        try:
            # prefer storing in config dict
            if isinstance(provider_instance.config, dict):
                provider_instance.config.setdefault("provider", provider_name)
            else:
                setattr(provider_instance, "provider", provider_name)
        except Exception:
            # best-effort - do not fail provider creation on tagging issues
            try:
                setattr(provider_instance, "provider", provider_name)
            except Exception:
                pass

        # Cache the instance if caching is enabled
        # if enable_cache:
        #     cls._search_provider_cache[provider_name] = provider_instance

        return provider_instance
    
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
    def is_search_provider(cls, provider, provider_name: str) -> bool:
        """check whether a provider instance corresponds to a named search provider.

        Detection order:
        1. provider.config['provider'] or similar explicit tag
        2. provider.provider attribute
        3. duck-typing (Azure index_client.create_index)
        Returns True if the provider matches provider_name.
        """
        try:
            if provider is None:
                return False
            cfg = getattr(provider, "config", None)
            if isinstance(cfg, dict):
                prov = cfg.get("provider") or cfg.get("provider_name") or cfg.get("name")
                if prov and str(prov).lower() == provider_name.lower():
                    return True

            prov_attr = getattr(provider, "provider", None)
            if prov_attr and str(prov_attr).lower() == provider_name.lower():
                return True

            # Azure duck-typing: has index_client.create_index
            if provider_name.lower().startswith("azure") and hasattr(provider, "index_client") and hasattr(provider.index_client, "create_index"):
                return True
        except Exception:
            pass
        return False
    
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