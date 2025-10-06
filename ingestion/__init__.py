"""
Lively Video Ingestion Pipeline

This package provides keyframe extraction, CLIP embedding generation,
and Azure AI Search indexing for video content.
"""

from .processing import KeyframeExtractor, extract_keyframes_from_video, CLIPEmbeddingsGenerator
from .search import KeyframeSearchIndex
from .storage import BlobStorageManager
from .pipeline import IngestionPipeline, run_ingestion
from .core import (
    FrameMetadata,
    FrameEmbedding,
    ProcessingContext,
    KeyframeExtractionConfig,
    EmbeddingConfig,
    BlobStorageConfig,
    SearchIndexConfig,
    get_file_hash,
    get_media_folder
)
from . import auth
from . import core
from . import processing
from . import storage
from . import search

__version__ = "1.0.0"

__all__ = [
    # Main pipeline
    "IngestionPipeline",
    "run_ingestion",
    # Components
    "KeyframeExtractor",
    "CLIPEmbeddingsGenerator",
    "KeyframeSearchIndex",
    "BlobStorageManager",
    # Convenience functions
    "extract_keyframes_from_video",
    # Models
    "FrameMetadata",
    "FrameEmbedding",
    "ProcessingContext",
    # Configuration
    "KeyframeExtractionConfig",
    "EmbeddingConfig",
    "BlobStorageConfig",
    "SearchIndexConfig",
    # Utils
    "get_file_hash",
    "get_media_folder",
    # Modules
    "auth",
    "core",
    "processing",
    "storage",
    "search",
]
