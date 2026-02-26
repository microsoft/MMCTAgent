"""
Search Document Models

Pydantic models for search index documents.
"""

from datetime import datetime
from typing import List, Optional
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
    object_collection: str = Field(
        default="[]",
        description="JSON string array of object collection tracking all objects (people, objects, etc.) in the video segment",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
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
    url: str = Field(
        ...,
        searchable=False,
        filterable=True,
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

    # — Chapter timestamp fields —
    start_time: float = Field(
        default=0.0,
        description="Chapter start time in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    end_time: float = Field(
        default=0.0,
        description="Chapter end time in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
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
        default_factory=list,
        description="Vector embedding for semantic search",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    # Sprint 1: New optional fields for dense captioning
    temporal_events_json: Optional[str] = Field(
        None,
        description="JSON serialized temporal events for complex queries",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    ocr_data_json: Optional[str] = Field(
        None,
        description="JSON serialized OCR data",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    key_points_json: Optional[str] = Field(
        None,
        description="JSON serialized key points",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    event_count: int = Field(
        default=0,
        description="Number of temporal events in this chapter",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    has_ocr: bool = Field(
        default=False,
        description="Whether this chapter has OCR data",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )


class SynopsisIndexDocument(BaseModel):
    """Document for synopsis search index."""
    
    # — Primary key —
    video_id: str = Field(
        ...,
        description="Unique identifier for the video",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=True
    )
    
    # — Summary fields —
    one_sentence_summary: str = Field(
        ...,
        description="One-sentence summary of the video",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    short_summary: str = Field(
        ...,
        description="Short summary (2-3 sentences)",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    full_summary: str = Field(
        ...,
        description="Full summary in paragraph form",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    key_points_json: str = Field(
        ...,
        description="JSON serialized key points for storage",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    topics_covered: List[str] = Field(
        default_factory=list,
        description="List of topics covered in the video",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    
    # — Vector embedding —
    synopsis_vector: Optional[List[float]] = Field(
        None,
        description="Embedding vector for synopsis",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Timestamp —
    created_at: str = Field(
        ...,
        description="ISO timestamp when synopsis was created",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )


class TemporalEventIndexDocument(BaseModel):
    """Document for temporal events search index.
    
    Relationship fields are filterable indexed arrays, NOT JSON strings,
    enabling fast O(1) lookups via Azure AI Search filters.
    """
    
    # — Primary key —
    event_id: str = Field(
        ...,
        description="Unique identifier for the event",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=True
    )
    
    # — Video and chapter context —
    video_id: str = Field(
        ...,
        description="Video identifier this event belongs to",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    chapter_index: int = Field(
        ...,
        description="Chapter index within the video",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    
    # — Event details —
    description: str = Field(
        ...,
        description="Description of the event",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    start_timestamp: float = Field(
        ...,
        description="Start time of the event in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    end_timestamp: float = Field(
        ...,
        description="End time of the event in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    duration: float = Field(
        ...,
        description="Duration of the event in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    event_type: str = Field(
        ...,
        description="Type of event (action, dialogue, transition, state_change)",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    participants: List[str] = Field(
        default_factory=list,
        description="List of participants involved in the event",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    sequence_number: int = Field(
        ...,
        description="Sequence order of the event",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    
    # — Indexed relationship fields for fast queries —
    # Direct adjacency (for chain traversal)
    prev_event_id: Optional[str] = Field(
        None,
        description="Previous event ID in sequence (filterable)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    next_event_id: Optional[str] = Field(
        None,
        description="Next event ID in sequence (filterable)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # Semantic relationships (for context queries)
    precedes_event_ids: List[str] = Field(
        default_factory=list,
        description="Event IDs that this event precedes (filterable array)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    follows_event_ids: List[str] = Field(
        default_factory=list,
        description="Event IDs that this event follows (filterable array)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Embeddings —
    description_vector: Optional[List[float]] = Field(
        None,
        description="Embedding vector for event description",
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
        default_factory=list,
        description="CLIP embedding vector for frame",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )


class ObjectCollectionDocument(BaseModel):
    """Document model for combined object collection search index."""

    id: str = Field(
        ...,
        description="Unique object collection document ID",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=True
    )

    video_id: str = Field(
        ...,
        description="Video hash ID this object collection belongs to",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    url: str = Field(
        default="",
        description="URL of the video",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    object_collection: str = Field(
        default="[]",
        description="JSON string array of merged object collection containing all objects (people, objects, etc.) from the entire video",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    object_count: int = Field(
        default=0,
        description="Total number of unique objects in the collection",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )

    video_summary: str = Field(
        default="",
        description="Overall summary of the entire video",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    embeddings: List[float] = Field(
        default_factory=list,
        description="Vector embedding of the video summary for semantic search",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    video_duration: float = Field(
        default=0.0,
        description="Duration of the video in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )


# ============================================================
# Sprint 1: Graph Index Documents for Temporal Knowledge Graphs
# ============================================================


class GraphEventIndexDocument(BaseModel):
    """Index document for graph-based event search.
    
    Searchable index document for events stored in temporal knowledge graphs,
    supporting both vector similarity and metadata-based search.
    """
    
    # — Primary key —
    id: str = Field(
        ...,
        description="Unique event document ID",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=True
    )
    
    # — Video and context —
    video_id: str = Field(
        ...,
        description="Video identifier this event belongs to",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    chapter_index: Optional[int] = Field(
        None,
        description="Chapter index within the video",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    
    # — Event content —
    description: str = Field(
        ...,
        description="Description of the event",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    event_type: Optional[str] = Field(
        None,
        description="Type of event (action, dialogue, transition, state_change)",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    participants: List[str] = Field(
        default_factory=list,
        description="List of participants involved in the event",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    
    # — Temporal fields —
    timestamp: float = Field(
        ...,
        description="Start timestamp of the event in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    duration: Optional[float] = Field(
        None,
        description="Duration of the event in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    sequence_number: Optional[int] = Field(
        None,
        description="Sequence order of the event",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    
    # — Relationship fields —
    prev_event_id: Optional[str] = Field(
        None,
        description="Previous event ID in sequence",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    next_event_id: Optional[str] = Field(
        None,
        description="Next event ID in sequence",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    related_object_ids: List[str] = Field(
        default_factory=list,
        description="IDs of objects involved in this event",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Vector embedding —
    embedding_vector: Optional[List[float]] = Field(
        None,
        description="Vector embedding for semantic similarity search",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Metadata —
    metadata_json: Optional[str] = Field(
        None,
        description="JSON serialized additional metadata",
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )


class GraphObjectIndexDocument(BaseModel):
    """Index document for graph-based object search.
    
    Searchable index document for objects stored in temporal knowledge graphs,
    supporting both vector similarity and metadata-based search.
    """
    
    # — Primary key —
    id: str = Field(
        ...,
        description="Unique object document ID",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=True
    )
    
    # — Video context —
    video_id: str = Field(
        ...,
        description="Video identifier this object belongs to",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    
    # — Object identity —
    name: str = Field(
        ...,
        description="Name or label of the object",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    object_type: Optional[str] = Field(
        None,
        description="Type of object (person, item, animal, text, etc.)",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    
    # — Appearance and identity descriptions —
    appearance: List[str] = Field(
        default_factory=list,
        description="Visual appearance descriptions",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    identity: List[str] = Field(
        default_factory=list,
        description="Identity descriptions (type, category, role)",
        searchable=True,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Temporal fields —
    first_seen: Optional[float] = Field(
        None,
        description="Timestamp when object first appears in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    last_seen: Optional[float] = Field(
        None,
        description="Timestamp when object last appears in seconds",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    
    # — Relationship fields —
    related_event_ids: List[str] = Field(
        default_factory=list,
        description="IDs of events involving this object",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    related_object_ids: List[str] = Field(
        default_factory=list,
        description="IDs of related objects (e.g., interactions)",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Vector embedding —
    embedding_vector: Optional[List[float]] = Field(
        None,
        description="Vector embedding for semantic similarity search",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    
    # — Metadata —
    metadata_json: Optional[str] = Field(
        None,
        description="JSON serialized additional metadata",
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
