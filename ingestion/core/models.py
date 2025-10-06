"""
Data models for video ingestion pipeline.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class FrameMetadata:
    """Metadata for an extracted keyframe."""
    frame_number: int
    timestamp_seconds: float
    motion_score: float


@dataclass
class FrameEmbedding:
    """Container for frame metadata and its embedding."""
    frame_metadata: FrameMetadata
    clip_embedding: np.ndarray
    blob_url: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class ProcessingContext:
    """Context object to hold processing state for a single video."""
    video_id: str
    video_path: str
    keyframe_metadata: Optional[List[FrameMetadata]] = None
    frame_embeddings: Optional[List[FrameEmbedding]] = None
    blob_urls: Optional[List[str]] = None
    youtube_url: Optional[str] = None

    def __post_init__(self):
        if self.keyframe_metadata is None:
            self.keyframe_metadata = []
        if self.frame_embeddings is None:
            self.frame_embeddings = []
        if self.blob_urls is None:
            self.blob_urls = []
