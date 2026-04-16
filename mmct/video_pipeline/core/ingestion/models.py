from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import uuid
import time


class TranslationResponse(BaseModel):
    """
    This model ensures that responses from the translation service have the correct structure
    with a list of translated text segments and other linked details like timestamps, count etc.
    """

    model_config = ConfigDict(extra="forbid")
    translations: List[str] = Field(
        ...,
        description="List of translated text segments with other linked details like timetamps, count, if there is any. it is in the same order as the input text, do not remove any of the segments.",
    )


class SubjectVarietyResponse(BaseModel):
    """Pydantic model for validating responses from the _extract_subject_and_variety function.

    This model represents the structured output from subject and variety identification analysis,
    extracting key subject matter information from video transcripts.
    """

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(
        ...,
        description="Name of the main subject or item which is talked about in the video, or 'None' if not found",
    )
    variety_of_subject: str = Field(
        ...,
        description="Name of the specific variety or type of subject mentioned in the video, or 'None' if not found",
    )


class ObjectResponse(BaseModel):
    """Pydantic model representing a single object tracked in the video.

    An object can be a person, object, animal, or any other entity that appears
    consistently throughout the video and is relevant to the content.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Name of the object if known, otherwise a short descriptive identity (e.g., 'iPhone 15 Pro', 'red car', 'main presenter', 'golden retriever')",
    )
    appearance: List[str] = Field(
        ...,
        description="List of appearance descriptions for this object (e.g., visual characteristics, color, shape, distinctive features)",
    )
    identity: List[str] = Field(
        ...,
        description="List of identity descriptions for this object (e.g., type, category, role, model number, brand, purpose)",
    )
    first_seen: float = Field(
        ..., description="Timestamp in seconds when this object first appears in the video"
    )
    additional_details: Optional[str] = Field(
        None,
        description="Any additional relevant information about this object that doesn't fit into the other categories (e.g., behavior, context, interactions, unique observations)",
    )


class ObjectCollection(BaseModel):
    """Pydantic model for the collection of all objects tracked in a video segment.

    This model maintains a collection of objects (people, objects, animals, etc.)
    identified and tracked throughout the video.
    """

    model_config = ConfigDict(extra="forbid")

    objects: Optional[List[ObjectResponse]] = Field(
        ...,
        description="List of ObjectResponse objects containing details like appearance, identity, and first appearance timestamp for each object (e.g., 'iPhone 15 Pro', 'main presenter', 'red car')",
    )

class SceneComposition(BaseModel):
    """Environment and spatial details for a video scene.
    
    Captures where the scene takes place, lighting conditions,
    camera framing, and spatial arrangement of elements.
    """

    model_config = ConfigDict(extra="forbid")

    environment_type: str = Field(
        ..., 
        description="Where the scene takes place",
        json_schema_extra={
            "examples": ["indoor", "outdoor", "studio", "vehicle", "kitchen", "office", "street", "park"]
        }
    )
    lighting_conditions: Optional[str] = Field(
        None, 
        description="How the scene is lit",
        json_schema_extra={
            "examples": ["natural", "artificial", "mixed", "dim", "bright", "low-light", "natural daylight"]
        }
    )
    camera_angle: Optional[str] = Field(
        None, 
        description="How the scene is framed/camera perspective",
        json_schema_extra={
            "examples": ["wide", "medium", "close-up", "overhead", "eye-level", "top-down", "establishing shot"]
        }
    )
    spatial_regions: Optional[Dict[str, List[str]]] = Field(
        default=None, 
        description="Mapping of spatial regions (foreground, background, left, right) to objects/people present in each area",
        json_schema_extra={
            "examples": [{"foreground": ["person in blue shirt", "laptop"], "background": ["bookshelf", "window"], "left": ["desk lamp"], "right": ["coffee mug"]}]
        }
    )

class ExtractedText(BaseModel):
    """Structured OCR extraction with categorization.
    
    Represents a single piece of text extracted from the video frames,
    with its type, position, and timing information.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The actual text content extracted from the frame")
    text_type: Literal["sign", "label", "caption", "title", "subtitle", "watermark", "ui_element"] = Field(
        ..., 
        description="Category of text: sign (fixed signage), label (product/name tags), caption (text overlays), title (headings), subtitle (CC text), watermark (logos), ui_element (buttons/menus)"
    )
    position: Literal["top-left", "top-center", "top-right", "center-left", "center", "center-right", "bottom-left", "bottom-center", "bottom-right"] = Field(
        ..., 
        description="Location of text in the frame"
    )
    confidence: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for the extraction (0.0 to 1.0)"
    )
    first_seen_timestamp: float = Field(
        ..., 
        ge=0.0,
        description="Timestamp in seconds when this text first appears"
    )


class ChapterOCRData(BaseModel):
    """Aggregated chapter-level OCR data.
    
    Contains all text extractions from a chapter, categorized by
    whether they persist throughout or appear briefly.
    """

    model_config = ConfigDict(extra="forbid")

    extractions: List[ExtractedText] = Field(
        default_factory=list, 
        description="List of all individual text extractions found in the chapter"
    )
    persistent_text: List[str] = Field(
        default_factory=list, 
        description="Text strings that appear throughout most of the chapter (e.g., watermarks, channel names)"
    )
    transient_text: List[str] = Field(
        default_factory=list, 
        description="Text strings that appear briefly (e.g., pop-up labels, temporary overlays)"
    )

class LLMConstraints(BaseModel):
    """Deployment constraints for adaptive extraction."""

    rate_limit_rpm: int = Field(default=60, description="Rate limit in requests per minute")
    rate_limit_tpm: int = Field(default=100000, description="Rate limit in tokens per minute")
    max_context_tokens: int = Field(default=128000, description="Maximum context tokens")
    max_output_tokens: int = Field(default=4096, description="Maximum output tokens")
    supports_vision: bool = Field(default=True, description="Whether model supports vision")
    supports_json_mode: bool = Field(default=True, description="Whether model supports JSON mode")

    @classmethod
    def from_config(cls, config: Dict) -> "LLMConstraints":
        """Create LLMConstraints from a configuration dictionary."""
        return cls(**config)


class ExtractionPlan(BaseModel):
    """Computed extraction strategy with error recovery."""

    # Core extraction parameters
    frames_per_chapter: int = Field(..., description="Number of frames to extract per chapter")
    use_unified_extraction: bool = Field(
        ..., description="Whether to use unified extraction approach"
    )
    concurrent_requests: int = Field(..., description="Number of concurrent requests allowed")
    batch_size: int = Field(..., description="Batch size for processing")
    estimated_tokens_per_chapter: int = Field(
        ..., description="Estimated tokens required per chapter"
    )

    # Error recovery configuration
    fallback_to_basic_on_failure: bool = Field(
        default=True, description="Fallback to basic extraction on failure"
    )
    max_retries_per_chapter: int = Field(
        default=3, description="Maximum retries per chapter"
    )
    partial_success_threshold: float = Field(
        default=0.7, description="Accept if 70% of chapters succeed"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, description="Number of failures before circuit breaker opens"
    )
    circuit_breaker_reset_seconds: int = Field(
        default=60, description="Seconds before circuit breaker resets"
    )


class ExtractionCircuitBreaker:
    """
    Prevents cascading failures during LLM issues.
    Opens circuit after consecutive failures, resets after timeout.
    """

    def __init__(self, failure_threshold: int = 5, reset_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_success(self):
        """Record a successful operation."""
        self.failures = 0
        self.is_open = False

    def record_failure(self):
        """Record a failed operation."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if not self.is_open:
            return True
        # Check if reset timeout has passed
        if self.last_failure_time and time.time() - self.last_failure_time > self.reset_seconds:
            self.is_open = False
            self.failures = 0
            return True
        return False


class ChapterResponse(BaseModel):
    """Chapter extraction response for video analysis.
    
    Extracts core chapter information in a single LLM call:
    - Timestamped description: primary output with [Xs] markers for citations
    - Scene composition (environment, lighting, camera, spatial layout)
    - OCR data (all visible text with categorization)
    
    The summary field is derived by stripping timestamps from the description.
    
    Note: Temporal events, visual actions, and objects are extracted 
    separately by the temporal_graph step.
    """
    
    model_config = ConfigDict(extra="forbid")

    timestamped_description: str = Field(
        ..., 
        description="Timestamped line-by-line description of the chapter. Each line starts with [Xs] (integer seconds from video start) followed by a readable sentence. Cover ALL visual and verbal content. No filler phrases."
    )
    scene_composition: Optional[SceneComposition] = Field(
        None, 
        description="Visual scene details including environment type, lighting conditions, camera angle, and spatial arrangement of elements in the frame"
    )
    ocr_data: Optional[ChapterOCRData] = Field(
        None, 
        description="All visible text extracted from keyframes, categorized by type (sign, label, caption, etc.) and persistence"
    )

    @property
    def summary(self) -> str:
        """Derive plain summary by stripping [Xs] timestamp markers."""
        from mmct.utils.timestamps import strip_timestamps
        return strip_timestamps(self.timestamped_description)
    
    @classmethod
    def get_extraction_guidelines(cls) -> str:
        """Return extraction guidelines for LLM prompts."""
        return """
### Timestamped Description Guidelines
- Each line starts with [Xs] where X is integer seconds from the video start
- Use the frame timestamps to assign correct [Xs] values (chronological order)
- Cover ALL visual and verbal content — zero information loss
- Write naturally readable sentences — not telegraphic or clipped
- Remove filler phrases: "the instructor", "the video shows", "viewers learn", "this section covers"
- Keep all technical terms, formulas, measurements
- If multiple related points share the same timestamp, combine into one flowing sentence
- Be specific about actions and outcomes

### Scene Composition Guidelines
- environment_type: Where the scene takes place (kitchen, office, outdoors, etc.)
- lighting_conditions: How the scene is lit (natural daylight, artificial, low-light, etc.)
- camera_angle: How the scene is framed (wide shot, medium shot, close-up, etc.)
- spatial_regions: Map foreground/background/left/right to objects/people present

### OCR Data Guidelines
- Extract ALL visible text from any source
- Categorize each text by type:
  - "sign": Fixed signage, labels on locations
  - "label": Product labels, name tags
  - "caption": Added text overlays, explanatory text
  - "title": Chapter/section titles, headings
  - "subtitle": Translated/closed caption text
  - "watermark": Logo/branding text
  - "ui_element": Buttons, menus, interface text
- Track which text persists throughout vs appears briefly
"""


# ============================================================
# Updated Existing Models
# ============================================================


class ChapterCreationResponse(BaseModel):
    """Pydantic model for validating responses from the create_chapter function.

    This model represents the structured output from video analysis, including
    detailed summary of content and object tracking.
    """

    model_config = ConfigDict(extra="forbid")

    detailed_summary: str = Field(
        ..., description="Comprehensive summary of the video content including frame analysis"
    )
    action_taken: Optional[str] = Field(
        None, description="Actions performed or demonstrated in the video"
    )
    text_from_scene: Optional[str] = Field(None, description="Text extracted from the video scenes")
    object_collection: Optional[List[ObjectResponse]] = Field(
        default=None,
        description="Collection of all objects (people, objects, animals, etc.) tracked in this video segment.",
    )

    def __str__(self, transcript: str = None) -> str:
        """


# Backwards compatible alias (old name)
DenseChapterResponse = ChapterResponse
        Generate a human-readable string representation of the chapter information
        formatted in natural language for creating text embeddings.

        Args:
            transcript (str, optional): The transcript text to add to the string representation.
                                       If not provided, transcript won't be included.

        Returns:
            str: Natural language representation of the chapter
        """
        # Start with the detailed summary
        text = f"{self.detailed_summary} "

        # Add actions if available
        if self.action_taken and self.action_taken.lower() != "none":
            text += f"The following actions are demonstrated in the video: {self.action_taken}. "

        # Add text from scene if available
        if self.text_from_scene and self.text_from_scene.lower() != "none":
            text += f"Text visible in the video includes: {self.text_from_scene}. "

        # Add object collection information if available
        if self.object_collection:
            text += "Objects in the video: "
            object_descriptions = []
            for object_info in self.object_collection:
                object_desc = f"{object_info.name} (first seen at {object_info.first_seen}s)"
                object_descriptions.append(object_desc)
            text += ", ".join(object_descriptions) + ". "

        # Add transcript if provided
        if transcript:
            text += f"The complete transcript of the video is as follows: {transcript}"

        return text


# ============================================================
# Merged Object Collection Response
# ============================================================


class MergedObjectCollectionResponse(BaseModel):
    """
    Response model for merged object collection.
    """

    model_config = ConfigDict(extra="forbid")

    merged_objects: Optional[List[ObjectResponse]] = Field(
        default_factory=list,
        description="List of ObjectResponse objects containing name, appearance, identity, first_seen timestamp, and additional_details",
    )


# ============================================================
# Metadata Models for Local JSON Storage
# ============================================================


class KeyframeMetadata(BaseModel):
    """Metadata for a single keyframe."""

    keyframe_filename: str = Field(
        ..., description="Filename of the extracted keyframe (e.g., 'abc123_0000.jpg')"
    )
    timestamp_seconds: float = Field(..., description="Time position in video (seconds)")
    file_path: str = Field(
        ..., description="Absolute path to the keyframe image file on local filesystem"
    )
    motion_score: float = Field(..., description="Optical flow motion score")
    embeddings: Optional[List[float]] = Field(
        default=None, description="512-dimensional CLIP embeddings (populated in Phase 2)"
    )
    blob_url: Optional[str] = Field(
        default="", description="Blob storage URL for the frame image (populated in Phase 3)"
    )


class KeyframeMetadataCollection(BaseModel):
    """Collection of all keyframe metadata for a video."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique collection document ID"
    )
    video_id: str = Field(..., description="Unique hash ID for the video part")
    video_duration: float = Field(..., description="Duration of the video part in seconds")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when metadata was created",
    )
    keyframes: List[KeyframeMetadata] = Field(
        ..., description="List of all keyframe metadata objects"
    )


class ChapterMetadata(BaseModel):
    """Metadata for a single chapter."""

    topic_of_video: str = Field(..., description="What the video is about")
    detailed_summary: str = Field(..., description="Long-form summary of the video")
    action_taken: str = Field(..., description="Actions described in the video")
    text_from_scene: str = Field(..., description="On-screen text detected")
    chapter_transcript: str = Field(..., description="Full transcript of the chapter")
    category: str = Field(..., description="Primary category")
    sub_category: str = Field(..., description="Sub-category")
    object_collection: str = Field(
        default="[]", description="JSON string array of object collection"
    )
    blob_frames_folder_path: str = Field(..., description="Blob storage path for keyframes folder")
    start_time: float = Field(default=0.0, description="Chapter start time in seconds")
    end_time: float = Field(default=0.0, description="Chapter end time in seconds")
    embeddings: Optional[List[float]] = Field(
        default=None, description="1536-dimensional text embeddings (populated in Phase 2)"
    )


class ChapterMetadataCollection(BaseModel):
    """Collection of all chapter metadata for a video."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique collection document ID"
    )
    hash_video_id: str = Field(..., description="Hash-based video identifier")
    video_duration: str = Field(
        default="None", description="Duration of this specific video part in seconds"
    )
    url: str = Field(..., description="URL to the video content")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when metadata was created",
    )
    chapters: List[ChapterMetadata] = Field(..., description="List of all chapter metadata objects")


class ObjectCollectionMetadata(BaseModel):
    """Metadata for the merged object collection and video summary."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique object collection document ID",
    )
    video_id: str = Field(..., description="Video hash ID this object collection belongs to")
    url: str = Field(default="", description="URL of the video")
    object_collection: str = Field(
        default="[]", description="JSON string array of merged object collection"
    )
    object_count: int = Field(
        default=0, description="Total number of unique objects in the collection"
    )
    video_summary: str = Field(default="", description="Overall summary of the entire video")
    embeddings: Optional[List[float]] = Field(
        default=None,
        description="1536-dimensional embeddings for video summary (populated in Phase 2)",
    )
    video_duration: float = Field(default=0.0, description="Duration of the video in seconds")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when metadata was created",
    )
    
# ============================================================
# Graph-based Models for Temporal Knowledge Graphs
# ============================================================


class GraphEvent(BaseModel):
    """Event node model for temporal knowledge graphs.
    
    Represents an event in a video with temporal and semantic properties
    for graph-based storage and retrieval.
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the event"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the event"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this event belongs to"
    )
    timestamp: Optional[float] = Field(
        default=None,
        description="Start timestamp of the event in seconds"
    )
    duration: Optional[float] = Field(
        default=None,
        description="Duration of the event in seconds"
    )
    event_type: Optional[str] = Field(
        default=None,
        description="Type of event (action, dialogue, transition, state_change)"
    )
    participants: Optional[List[str]] = Field(
        default=None,
        description="List of participants involved in the event"
    )
    chapter_index: Optional[int] = Field(
        default=None,
        description="Chapter index within the video"
    )
    sequence_number: Optional[int] = Field(
        default=None,
        description="Sequence order of the event"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity search"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the event"
    )


class GraphObject(BaseModel):
    """Object node model for temporal knowledge graphs.
    
    Represents a tracked object (person, item, entity) in a video
    for graph-based storage and retrieval.
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the object"
    )
    name: Optional[str] = Field(
        default=None,
        description="Name or label of the object"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this object belongs to"
    )
    first_seen: Optional[float] = Field(
        default=None,
        description="Timestamp when object first appears in seconds"
    )
    last_seen: Optional[float] = Field(
        default=None,
        description="Timestamp when object last appears in seconds"
    )
    object_type: Optional[str] = Field(
        default=None,
        description="Type of object (person, item, animal, text, etc.)"
    )
    appearance: Optional[List[str]] = Field(
        default=None,
        description="Visual appearance descriptions"
    )
    identity: Optional[List[str]] = Field(
        default=None,
        description="Identity descriptions (type, category, role)"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity search"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the object"
    )
    # Canonical object tracking for deduplication
    canonical_id: Optional[str] = Field(
        default=None,
        description="ID of the canonical object this was merged into"
    )
    is_canonical: bool = Field(
        default=True,
        description="Whether this object is the canonical representation"
    )
    merged_from: Optional[List[str]] = Field(
        default=None,
        description="List of object IDs merged into this canonical"
    )
    appearance_count: int = Field(
        default=1,
        description="Number of times this object appeared"
    )


class GraphChapter(BaseModel):
    """Chapter node model for temporal knowledge graphs.
    
    Represents a video chapter/segment with temporal boundaries and content
    summary for hierarchical graph organization.
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the chapter"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this chapter belongs to"
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index of the chapter within the video (0-indexed)"
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start timestamp of the chapter in seconds"
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End timestamp of the chapter in seconds"
    )
    video_duration: Optional[float] = Field(
        default=None,
        description="Total duration of the video in seconds"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of the chapter content (legacy; prefer timestamped_description)"
    )
    timestamped_description: Optional[str] = Field(
        default=None,
        description="Timestamped line-by-line description ([Xs] sentence format)"
    )
    scene_composition: Optional[str] = Field(
        default=None,
        description="Human-readable scene description (environment, lighting, camera angle)"
    )
    ocr_data: Optional[str] = Field(
        default=None,
        description="Human-readable visible text extracted from keyframes"
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Transcript text of the chapter"
    )
    group_index: Optional[int] = Field(
        default=None,
        description="Index of the chapter group this chapter belongs to"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity search"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the chapter"
    )


class GraphChapterGroup(BaseModel):
    """Chapter group node model for temporal knowledge graphs.
    
    Represents a logical grouping of chapters for hierarchical video organization
    and high-level topic navigation.
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the chapter group"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this group belongs to"
    )
    name: Optional[str] = Field(
        default=None,
        description="Name or title of the chapter group"
    )
    order: Optional[int] = Field(
        default=None,
        description="Sequential order of group within video (0-indexed)"
    )
    chapter_indices: Optional[List[int]] = Field(
        default=None,
        description="Indices of chapters belonging to this group"
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start timestamp of the group in seconds"
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End timestamp of the group in seconds"
    )
    video_duration: Optional[float] = Field(
        default=None,
        description="Total duration of the video in seconds"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of the chapter group content"
    )
    topics: Optional[List[str]] = Field(
        default=None,
        description="Topics covered in this chapter group"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic similarity search"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the group"
    )


class GraphKeyframe(BaseModel):
    """Keyframe node model for temporal knowledge graphs.
    
    Represents a keyframe image extracted from a video chapter with
    visual embedding for image-text similarity search.
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the keyframe"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this keyframe belongs to"
    )
    chapter_id: Optional[str] = Field(
        default=None,
        description="Chapter identifier this keyframe belongs to"
    )
    timestamp: Optional[float] = Field(
        default=None,
        description="Timestamp of the keyframe in seconds"
    )
    frame_index: Optional[int] = Field(
        default=None,
        description="Index of the frame within the chunk"
    )
    filepath: Optional[str] = Field(
        default=None,
        description="Local file path to the keyframe image"
    )
    blob_name: Optional[str] = Field(
        default=None,
        description="Blob storage name for the uploaded keyframe"
    )
    blob_url: Optional[str] = Field(
        default=None,
        description="Blob storage URL for the uploaded keyframe"
    )
    motion_score: Optional[float] = Field(
        default=None,
        description="Motion score indicating visual activity level"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="512-dim CLIP embedding for image-text similarity search"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the keyframe"
    )


class GraphTranscript(BaseModel):
    """Transcript node model for temporal knowledge graphs.
    
    Represents the raw verbal content of a video chapter, separate from
    the multimodal summary in GraphChapter. This enables targeted retrieval
    for verbal-only queries without visual context overhead.
    
    Use Cases:
    - Quote search ("What did the speaker say about X?")
    - Spoken content lookup
    - Dialogue analysis
    - When visual context is explicitly NOT needed
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the transcript (format: transcript_{video_id}_{chunk_index:03d})"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this transcript belongs to"
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index of the chapter/chunk this transcript corresponds to (0-indexed)"
    )
    transcript: Optional[str] = Field(
        default=None,
        description="Raw transcript text (verbal content only)"
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start timestamp of the transcript segment in seconds"
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End timestamp of the transcript segment in seconds"
    )
    video_duration: Optional[float] = Field(
        default=None,
        description="Total duration of the video in seconds"
    )
    embedding_vector: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding (BGE-small 384-dim) for semantic search"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the transcript"
    )

class ChapterGroup(BaseModel):
    """Grouping model for organizing related chapters.
    
    Represents a logical grouping of chapters based on topic,
    timeline, or other criteria for hierarchical video organization.
    """
    
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the chapter group"
    )
    name: Optional[str] = Field(
        default=None,
        description="Name or title of the chapter group"
    )
    video_id: Optional[str] = Field(
        default=None,
        description="Video identifier this group belongs to"
    )
    order: Optional[int] = Field(
        default=None,
        description="Sequential order of group within video (0-indexed, for sorting/navigation)"
    )
    chapter_indices: Optional[List[int]] = Field(
        default=None,
        description="Indices of chapters in this group"
    )
    start_time: Optional[float] = Field(
        default=None,
        description="Start timestamp of the group in seconds"
    )
    end_time: Optional[float] = Field(
        default=None,
        description="End timestamp of the group in seconds"
    )
    video_duration: Optional[float] = Field(
        default=None,
        description="Total duration of the video in seconds"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of the chapter group content"
    )
    topics: Optional[List[str]] = Field(
        default=None,
        description="Topics covered in this chapter group"
    )
    parent_group_id: Optional[str] = Field(
        default=None,
        description="Parent group ID for hierarchical organization"
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Additional metadata for the group"
    )
