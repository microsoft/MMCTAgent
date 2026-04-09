"""
Built-in pipeline steps for video ingestion.

Importing this module registers all built-in steps with the framework.
"""

# Import all step modules to trigger @register_step decorators

from .transcription_step import TranscriptionStep

from .graph_validation.step import GraphValidationStep
from .compress.step import CompressionStep
from .transcript_upload.step import TranscriptUploadStep
from .video_chunking.step import VideoChunkingStep
from .extraction_planning.step import ExtractionPlanningStep
from .uniform_frames.step import UniformFrameExtractionStep

# Backwards compatible step types are still registered, but codebase naming prefers non-prefixed names.
from .keyframes.step import KeyframeExtractionStep
from .chapters.step import ChapterGenerationStep

from .temporal_graph.step import TemporalGraphStep
from .chapter_grouping.step import ChapterGroupingStep
from .keyframe_upload.step import KeyframeUploadStep
from .graph_construction.step import GraphConstructionStep
from .graph_upload.step import GraphUploadStep

from .export.step import ExportStep
from .cleanup.step import CleanupStep

__all__ = [
    "TranscriptionStep",
    "GraphValidationStep",
    "CompressionStep",
    "TranscriptUploadStep",
    "VideoChunkingStep",
    "ExtractionPlanningStep",
    "UniformFrameExtractionStep",
    "KeyframeExtractionStep",
    "ChapterGenerationStep",
    "TemporalGraphStep",
    "ChapterGroupingStep",
    "GraphConstructionStep",
    "GraphUploadStep",
    "KeyframeUploadStep",
    "ExportStep",
    "CleanupStep",
]
