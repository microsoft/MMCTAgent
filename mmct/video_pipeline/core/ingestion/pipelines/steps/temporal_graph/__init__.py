"""Temporal graph extraction pipeline step.

Extracts events and objects from video chapters for temporal knowledge graph construction.

Components:
- TemporalGraphStep: Main pipeline step with parallel processing
- EventExtractor: Multimodal event extraction (frames + transcript)
- ObjectExtractor: Visual-only object extraction with batch deduplication

Embedding Models (fastembed, CPU-optimized):
- Events: BAAI/bge-small-en-v1.5 (384-dim) - semantic similarity
- Objects: snowflake/snowflake-arctic-embed-s (384-dim) - deduplication

Provider implementations are in mmct.providers.custom_providers:
- FastEmbedBGEsmallEmbeddingProvider
- FastEmbedArcticEmbeddingProvider
"""

from .step import TemporalGraphStep
from .event_extractor import EventExtractor
from .object_extractor import ObjectExtractor
from .local_embeddings import (
    get_event_embedding_provider,
    get_object_embedding_provider,
    cleanup_embedding_providers,
)

# Re-export the providers from custom_providers for convenience
from mmct.providers.custom_providers import (
    FastEmbedBGEsmallEmbeddingProvider,
    FastEmbedArcticEmbeddingProvider,
)

__all__ = [
    "TemporalGraphStep",
    "EventExtractor",
    "ObjectExtractor",
    "FastEmbedBGEsmallEmbeddingProvider",
    "FastEmbedArcticEmbeddingProvider",
    "get_event_embedding_provider",
    "get_object_embedding_provider",
    "cleanup_embedding_providers",
]
