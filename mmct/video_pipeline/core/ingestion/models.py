from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
import uuid

class TranslationResponse(BaseModel):
    """
    This model ensures that responses from the translation service have the correct structure
    with a list of translated text segments and other linked details like timestamps, count etc.
    """
    model_config = ConfigDict(extra="forbid")
    translations: List[str] = Field(
        ..., 
        description="List of translated text segments with other linked details like timetamps, count, if there is any. it is in the same order as the input text, do not remove any of the segments."
    )


class SubjectVarietyResponse(BaseModel):
    """Pydantic model for validating responses from the _extract_subject_and_variety function.
    
    This model represents the structured output from subject and variety identification analysis,
    extracting key subject matter information from video transcripts.
    """
    model_config = ConfigDict(extra="forbid")
    
    subject: str = Field(
        ..., 
        description="Name of the main subject or item which is talked about in the video, or 'None' if not found"
    )
    variety_of_subject: str = Field(
        ..., 
        description="Name of the specific variety or type of subject mentioned in the video, or 'None' if not found"
    )

class ObjectResponse(BaseModel):
    """Pydantic model representing a single object tracked in the video.

    An object can be a person, object, animal, or any other entity that appears
    consistently throughout the video and is relevant to the content.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Name of the object if known, otherwise a short descriptive identity (e.g., 'iPhone 15 Pro', 'red car', 'main presenter', 'golden retriever')"
    )
    appearance: List[str] = Field(
        ...,
        description="List of appearance descriptions for this object (e.g., visual characteristics, color, shape, distinctive features)"
    )
    identity: List[str] = Field(
        ...,
        description="List of identity descriptions for this object (e.g., type, category, role, model number, brand, purpose)"
    )
    first_seen: float = Field(
        ...,
        description="Timestamp in seconds when this object first appears in the video"
    )
    additional_details: Optional[str] = Field(
        None,
        description="Any additional relevant information about this object that doesn't fit into the other categories (e.g., behavior, context, interactions, unique observations)"
    )


class ObjectCollection(BaseModel):
    """Pydantic model for the collection of all objects tracked in a video segment.

    This model maintains a collection of objects (people, objects, animals, etc.)
    identified and tracked throughout the video.
    """
    model_config = ConfigDict(extra="forbid")

    objects: Optional[List[ObjectResponse]] = Field(
        ...,
        description="List of ObjectResponse objects containing details like appearance, identity, and first appearance timestamp for each object (e.g., 'iPhone 15 Pro', 'main presenter', 'red car')"
    )



class ChapterCreationResponse(BaseModel):
    """Pydantic model for validating responses from the create_chapter function.

    This model represents the structured output from video analysis, including
    detailed summary of content and object tracking.
    """
    model_config = ConfigDict(extra="forbid")

    detailed_summary: str = Field(
        ...,
        description="Comprehensive summary of the video content including frame analysis"
    )
    action_taken: Optional[str] = Field(
        None,
        description="Actions performed or demonstrated in the video"
    )
    text_from_scene: Optional[str] = Field(
        None,
        description="Text extracted from the video scenes"
    )
    object_collection: Optional[List[ObjectResponse]] = Field(
        default=None,
        description="Collection of all objects (people, objects, animals, etc.) tracked in this video segment."
    )
    
    def __str__(self, transcript: str = None) -> str:
        """
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
# Metadata Models for Local JSON Storage
# ============================================================

class KeyframeMetadata(BaseModel):
    """Metadata for a single keyframe."""

    keyframe_filename: str = Field(
        ...,
        description="Filename of the extracted keyframe (e.g., 'abc123_0000.jpg')"
    )
    timestamp_seconds: float = Field(
        ...,
        description="Time position in video (seconds)"
    )
    file_path: str = Field(
        ...,
        description="Absolute path to the keyframe image file on local filesystem"
    )
    motion_score: float = Field(
        ...,
        description="Optical flow motion score"
    )
    embeddings: Optional[List[float]] = Field(
        default=None,
        description="512-dimensional CLIP embeddings (populated in Phase 2)"
    )
    blob_url: Optional[str] = Field(
        default="",
        description="Blob storage URL for the frame image (populated in Phase 3)"
    )


class KeyframeMetadataCollection(BaseModel):
    """Collection of all keyframe metadata for a video."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique collection document ID"
    )
    video_id: str = Field(
        ...,
        description="Unique hash ID for the video part"
    )
    parent_id: str = Field(
        ...,
        description="Hash ID of the original video before splitting"
    )
    video_duration: float = Field(
        ...,
        description="Duration of the video part in seconds"
    )
    parent_duration: float = Field(
        ...,
        description="Duration of the original video in seconds"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when metadata was created"
    )
    keyframes: List[KeyframeMetadata] = Field(
        ...,
        description="List of all keyframe metadata objects"
    )


class ChapterMetadata(BaseModel):
    """Metadata for a single chapter."""

    topic_of_video: str = Field(
        ...,
        description="What the video is about"
    )
    detailed_summary: str = Field(
        ...,
        description="Long-form summary of the video"
    )
    action_taken: str = Field(
        ...,
        description="Actions described in the video"
    )
    text_from_scene: str = Field(
        ...,
        description="On-screen text detected"
    )
    chapter_transcript: str = Field(
        ...,
        description="Full transcript of the chapter"
    )
    category: str = Field(
        ...,
        description="Primary category"
    )
    sub_category: str = Field(
        ...,
        description="Sub-category"
    )
    object_collection: str = Field(
        default="[]",
        description="JSON string array of object collection"
    )
    blob_frames_folder_path: str = Field(
        ...,
        description="Blob storage path for keyframes folder"
    )
    start_time: float = Field(
        default=0.0,
        description="Chapter start time in seconds"
    )
    end_time: float = Field(
        default=0.0,
        description="Chapter end time in seconds"
    )
    embeddings: Optional[List[float]] = Field(
        default=None,
        description="1536-dimensional text embeddings (populated in Phase 2)"
    )


class ChapterMetadataCollection(BaseModel):
    """Collection of all chapter metadata for a video."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique collection document ID"
    )
    hash_video_id: str = Field(
        ...,
        description="Hash-based video identifier"
    )
    parent_id: str = Field(
        default="None",
        description="Original video ID (before splitting)"
    )
    parent_duration: str = Field(
        default="None",
        description="Original video duration in seconds"
    )
    video_duration: str = Field(
        default="None",
        description="Duration of this specific video part in seconds"
    )
    url: str = Field(
        ...,
        description="URL to the video content"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when metadata was created"
    )
    chapters: List[ChapterMetadata] = Field(
        ...,
        description="List of all chapter metadata objects"
    )


class ObjectCollectionMetadata(BaseModel):
    """Metadata for the merged object collection and video summary."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique object collection document ID"
    )
    video_id: str = Field(
        ...,
        description="Video hash ID this object collection belongs to"
    )
    url: str = Field(
        default="",
        description="URL of the video"
    )
    object_collection: str = Field(
        default="[]",
        description="JSON string array of merged object collection"
    )
    object_count: int = Field(
        default=0,
        description="Total number of unique objects in the collection"
    )
    video_summary: str = Field(
        default="",
        description="Overall summary of the entire video"
    )
    embeddings: Optional[List[float]] = Field(
        default=None,
        description="1536-dimensional embeddings for video summary (populated in Phase 2)"
    )
    video_duration: float = Field(
        default=0.0,
        description="Duration of the video in seconds"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="ISO 8601 timestamp when metadata was created"
    )