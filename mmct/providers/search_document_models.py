"""
Search Document Models

Pydantic models for search index documents.
"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class ChapterIndexDocument(BaseModel):
    """Document model for video chapter search index."""

    # — Primary key —
    id: str = Field(
        ...,
        description="Unique document ID",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=True
    )

    # — Searchable text fields —
    topic_of_video: str = Field(
        ...,
        description="What the video is about",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    detailed_summary: str = Field(
        ...,
        description="Long-form summary of the video",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    action_taken: str = Field(
        ...,
        description="Actions described in the video",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    text_from_scene: str = Field(
        ...,
        description="On-screen text detected",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    chapter_transcript: str = Field(
        ...,
        description="Full transcript of the chapter",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    category: str = Field(
        ...,
        description="Primary category",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    sub_category: str = Field(
        ...,
        description="Sub-category",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    subject: str = Field(
        ...,
        description="Main subject or item mentioned",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    variety: str = Field(
        ...,
        description="Variety or type of subject",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    hash_video_id: str = Field(
        ...,
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    parent_id: str = Field(
        default="None",
        description="Original video ID (before splitting)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    parent_duration: str = Field(
        default="None",
        description="Original video duration in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    video_duration: str = Field(
        default="None",
        description="Duration of this specific video part in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    # — Non-searchable metadata —
    youtube_url: str = Field(
        ...,
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_video_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_audio_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_transcript_file_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_frames_folder_path: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    # — Date & vector fields —
    time: datetime = Field(
        ...,
        description="Ingestion timestamp",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    embeddings: List[float] = Field(
        ...,
        description="Vector embedding for semantic search",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )


class KeyframeDocument(BaseModel):
    """Document model for keyframe/frame search index."""

    # — Primary key —
    id: str = Field(
        ...,
        description="Unique frame document ID",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=True
    )

    # — Metadata fields —
    video_id: str = Field(
        ...,
        description="Hash-based video identifier",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    keyframe_filename: str = Field(
        ...,
        description="Filename of the extracted keyframe",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    created_at: datetime = Field(
        ...,
        description="Frame extraction timestamp",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    motion_score: float = Field(
        ...,
        description="Optical flow motion score",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    timestamp_seconds: float = Field(
        ...,
        description="Time position in video (seconds)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    blob_url: str = Field(
        default="",
        description="Blob storage URL for the frame image",
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    parent_id: str = Field(
        default="",
        description="Original video ID (before splitting)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    parent_duration: float = Field(
        default=0.0,
        description="Duration of parent video in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    video_duration: float = Field(
        default=0.0,
        description="Duration of this video part in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )

    # — Vector embedding field —
    embeddings: List[float] = Field(
        ...,
        description="CLIP embedding vector for frame",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
