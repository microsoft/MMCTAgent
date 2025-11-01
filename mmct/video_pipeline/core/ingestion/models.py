from typing import List, Optional
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


class ChapterCreationResponse(BaseModel):
    """Pydantic model for validating responses from the create_chapter function.
    
    This model represents the structured output from video analysis, including topic information,
    categorization, species identification, and detailed summary of content.
    """
    model_config = ConfigDict(extra="forbid")
    
    Topic_of_video: str = Field(
        ..., 
        description="Main topic or theme that is discussed in the video"
    )
    Category: str = Field(
        ..., 
        description="The primary category the video content belongs to"
    )
    Sub_category: Optional[str] = Field(
        None, 
        description="The sub-category the video content belongs to"
    )
    subject: Optional[str] = Field(
        None, 
        description="Name of the main subject or item discussed in the video"
    )
    variety_of_subject: Optional[str] = Field(
        None, 
        description="Specific variety or type of subject mentioned in the video"
    )
    Detailed_summary: str = Field(
        ..., 
        description="Comprehensive summary of the video content including frame analysis"
    )
    Action_taken: Optional[str] = Field(
        None, 
        description="Actions performed or demonstrated in the video"
    )
    Text_from_scene: Optional[str] = Field(
        None, 
        description="Text extracted from the video scenes"
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
        # Start with the topic and category
        text = f"This video is about {self.Topic_of_video}. "
        text += f"It belongs to the {self.Category} category"
        
        # Add subcategory if available
        if self.Sub_category:
            text += f", specifically in the {self.Sub_category} subcategory"
        text += ". "
        
        # Add subject information if available
        if self.subject and self.subject.lower() != "none":
            text += f"The video discusses {self.subject}"
            if self.variety_of_subject and self.variety_of_subject.lower() != "none":
                text += f", particularly the {self.variety_of_subject} variety"
            text += ". "
        
        # Add the detailed summary
        text += f"{self.Detailed_summary} "
        
        # Add actions if available
        if self.Action_taken and self.Action_taken.lower() != "none":
            text += f"The following actions are demonstrated in the video: {self.Action_taken}. "
        
        # Add text from scene if available
        if self.Text_from_scene and self.Text_from_scene.lower() != "none":
            text += f"Text visible in the video includes: {self.Text_from_scene}. "
        
        # Add transcript if provided
        if transcript:
            text += f"The complete transcript of the video is as follows: {transcript}"
        
        return text