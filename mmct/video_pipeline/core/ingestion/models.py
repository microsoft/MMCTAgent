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


class SpeciesVarietyResponse(BaseModel):
    """Pydantic model for validating responses from the species_and_variety function.
    
    This model represents the structured output from species and variety identification analysis,
    extracting key agricultural information from video transcripts.
    """
    model_config = ConfigDict(extra="forbid")
    
    species: str = Field(
        ..., 
        description="Name of the species which is talked about in the video, or 'None' if not found"
    )
    Variety_of_species: str = Field(
        ..., 
        description="Name of the variety of species mentioned in the video, or 'None' if not found"
    )


class ChapterCreationResponse(BaseModel):
    """Pydantic model for validating responses from the Chapters_creation function.
    
    This model represents the structured output from video analysis, including topic information,
    categorization, species identification, and detailed summary of content.
    """
    model_config = ConfigDict(extra="forbid")
    
    Topic_of_video: str = Field(
        ..., 
        description="Topic related to agriculture that is discussed in the video"
    )
    Category: str = Field(
        ..., 
        description="The primary category the video content belongs to"
    )
    Sub_category: Optional[str] = Field(
        None, 
        description="The sub-category the video content belongs to"
    )
    species: Optional[str] = Field(
        None, 
        description="Name of the species discussed in the video"
    )
    Variety_of_species: Optional[str] = Field(
        None, 
        description="Specific variety of species mentioned in the video"
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
        
        # Add species information if available
        if self.species and self.species.lower() != "none":
            text += f"The video discusses the {self.species} species"
            if self.Variety_of_species and self.Variety_of_species.lower() != "none":
                text += f", particularly the {self.Variety_of_species} variety"
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