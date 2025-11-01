"""
Search Document Models

Pydantic models for search index documents.
"""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class AISearchDocument(BaseModel):
    """Document model for Azure AI Search video chapter index."""

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
