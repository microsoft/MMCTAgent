"""Core models, configuration, and utilities for the ingestion pipeline."""

from .models import FrameMetadata, FrameEmbedding, ProcessingContext
from .config import (
    KeyframeExtractionConfig,
    EmbeddingConfig,
    BlobStorageConfig,
    SearchIndexConfig
)
from .utils import get_file_hash, get_media_folder

__all__ = [
    "FrameMetadata",
    "FrameEmbedding",
    "ProcessingContext",
    "KeyframeExtractionConfig",
    "EmbeddingConfig",
    "BlobStorageConfig",
    "SearchIndexConfig",
    "get_file_hash",
    "get_media_folder",
]
