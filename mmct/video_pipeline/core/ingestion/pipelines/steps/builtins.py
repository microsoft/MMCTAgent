"""
Built-in pipeline steps for video ingestion.

Importing this module registers all built-in steps with the framework.
"""

# Import all step modules to trigger @register_step decorators

# Simple steps
from .pre_validation_step import PreValidationStep
from .transcription_step import TranscriptionStep

from .compress.step import CompressionStep
from .video_chunking.step import VideoChunkingStep
from .keyframes.step import KeyframeExtractionStep
from .chapters import ChapterGenerationStep
from .chapter_enrichment.step import ChapterEnrichmentStep
from .embeddings.step import EmbeddingGenerationStep
from .upload.step import UploadStep
from .cleanup.step import CleanupStep

__all__ = [
    "PreValidationStep",
    "TranscriptionStep",
    "CompressionStep",
    "VideoChunkingStep",
    "KeyframeExtractionStep",
    "ChapterGenerationStep",
    "ChapterEnrichmentStep",
    "EmbeddingGenerationStep",
    "UploadStep",
    "CleanupStep",
]
