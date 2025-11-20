from typing import List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

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