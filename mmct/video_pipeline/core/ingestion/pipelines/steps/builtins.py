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

# Extraction planning
from .extraction_planning import ExtractionPlanningStep

# Dense captioning steps
from .dense_keyframes import DenseKeyframeExtractionStep
from .dense_chapters import DenseChapterGenerationStep
from .dense_export import DenseExportStep

# Temporal graph steps
from .temporal_graph import TemporalGraphStep
from .chapter_grouping import ChapterGroupingStep
from .graph_construction import GraphConstructionStep
from .graph_upload import GraphUploadStep

# Upload steps
from .keyframe_upload import KeyframeUploadStep

# Uniform frame extraction
from .uniform_frames import UniformFrameExtractionStep

# Transcript upload
from .transcript_upload import TranscriptUploadStep

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
    # Extraction planning
    "ExtractionPlanningStep",
    # Dense captioning steps
    "DenseKeyframeExtractionStep",
    "DenseChapterGenerationStep",
    "DenseExportStep",
    # Temporal graph steps
    "TemporalGraphStep",
    "ChapterGroupingStep",
    "GraphConstructionStep",
    "GraphUploadStep",
    # Upload steps
    "KeyframeUploadStep",
    # Uniform frame extraction
    "UniformFrameExtractionStep",
    # Transcript upload
    "TranscriptUploadStep",
]
