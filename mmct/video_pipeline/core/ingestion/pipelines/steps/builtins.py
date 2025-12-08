"""
Built-in pipeline steps for video ingestion.

Importing this module registers all built-in steps with the framework.
"""

# Import all step modules to trigger @register_step decorators

# Simple steps
from .pre_validation_step import PreValidationStep
from .transcription_step import TranscriptionStep

# Complex steps (wrappers for existing modules)
from .compress.step import CompressionStep
from .keyframes.step import KeyframeExtractionStep
from .chapters.step import ChapterGenerationStep
from .embeddings.step import EmbeddingGenerationStep
from .upload.step import UploadStep
from .cleanup.step import CleanupStep

__all__ = [
    # Simple steps
    "PreValidationStep",
    "TranscriptionStep",
    # Complex steps
    "CompressionStep",
    "KeyframeExtractionStep",
    "ChapterGenerationStep",
    "EmbeddingGenerationStep",
    "UploadStep",
    "CleanupStep",
]
