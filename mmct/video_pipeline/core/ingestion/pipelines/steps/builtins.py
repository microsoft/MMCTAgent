"""
Built-in pipeline steps for video ingestion.

Importing this module registers all built-in steps with the framework.
"""

# Import all step modules to trigger @register_step decorators

# Core steps
from .transcription_step import TranscriptionStep
from .compress.step import CompressionStep
from .video_chunking.step import VideoChunkingStep
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
from .graph_validation import GraphValidationStep

# Upload steps
from .keyframe_upload import KeyframeUploadStep

# Uniform frame extraction
from .uniform_frames import UniformFrameExtractionStep

# Transcript upload
from .transcript_upload import TranscriptUploadStep

__all__ = [
    "TranscriptionStep",
    "CompressionStep",
    "VideoChunkingStep",
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
    "GraphValidationStep",
    # Upload steps
    "KeyframeUploadStep",
    # Uniform frame extraction
    "UniformFrameExtractionStep",
    # Transcript upload
    "TranscriptUploadStep",
]
