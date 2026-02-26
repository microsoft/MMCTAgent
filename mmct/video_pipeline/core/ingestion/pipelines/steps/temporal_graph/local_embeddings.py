"""Local embedding provider singletons for temporal graph step.

Provides singleton accessors for the fastembed providers:
- FastEmbedBGEsmallEmbeddingProvider: Uses BAAI/bge-small-en-v1.5 (384-dim)
- FastEmbedArcticEmbeddingProvider: Uses snowflake/snowflake-arctic-embed-s (384-dim)

The actual provider implementations are in mmct.providers.custom_providers.
"""

from typing import Optional

from loguru import logger

from mmct.providers.custom_providers import (
    FastEmbedBGEsmallEmbeddingProvider,
    FastEmbedArcticEmbeddingProvider,
)


# Singleton instances for reuse within a pipeline run
_event_provider: Optional[FastEmbedBGEsmallEmbeddingProvider] = None
_object_provider: Optional[FastEmbedArcticEmbeddingProvider] = None


def get_event_embedding_provider() -> FastEmbedBGEsmallEmbeddingProvider:
    """Get or create the singleton event embedding provider.
        
    Returns:
        FastEmbedBGEsmallEmbeddingProvider instance (BAAI/bge-small-en-v1.5, 384-dim)
    """
    global _event_provider
    if _event_provider is None:
        _event_provider = FastEmbedBGEsmallEmbeddingProvider()
    return _event_provider


def get_object_embedding_provider() -> FastEmbedArcticEmbeddingProvider:
    """Get or create the singleton object embedding provider.
        
    Returns:
        FastEmbedArcticEmbeddingProvider instance (snowflake-arctic-embed-s, 384-dim)
    """
    global _object_provider
    if _object_provider is None:
        _object_provider = FastEmbedArcticEmbeddingProvider()
    return _object_provider


def cleanup_embedding_providers() -> None:
    """Clean up all singleton embedding providers."""
    global _event_provider, _object_provider
    if _event_provider is not None:
        _event_provider.close()
        _event_provider = None
    if _object_provider is not None:
        _object_provider.close()
        _object_provider = None
    logger.info("Embedding providers cleaned up")
