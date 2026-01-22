"""
Pydantic models for V2 orchestrator responses with citation support.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TimestampRange(BaseModel):
    """A pair of start and end timestamps."""
    start_time: str = Field(..., description="Start timestamp in HH:MM:SS format")
    end_time: str = Field(..., description="End timestamp in HH:MM:SS format")


class CitationSource(BaseModel):
    """
    Source information for a single citation reference.
    Each citation maps to exactly ONE timestamp range from ONE video.
    """
    citation_id: int = Field(..., description="The citation number referenced in the response (e.g., 1 for [1])")
    video_id: str = Field(..., description="Hash video ID from context")
    url: str = Field(default="", description="YouTube URL, if available")
    start_time: str = Field(..., description="Start timestamp in HH:MM:SS format")
    end_time: str = Field(..., description="End timestamp in HH:MM:SS format")


class V2AgentResponse(BaseModel):
    """
    Pydantic model for V2 orchestrator responses with inline citations.
    
    The response text contains citations in square brackets [1], [2], etc.
    Each citation maps to exactly ONE timestamp range from a video.
    If multiple timestamp ranges are relevant, create multiple citations.
    """
    model_config = ConfigDict(extra="forbid")
    
    @staticmethod
    def get_schema_template() -> str:
        """Returns the JSON schema template for use in prompts."""
        return '''{
  "response": "<Markdown-formatted response with inline citations [1], [2], etc. Each citation = one timestamp range.>",
  "answer_found": true/false,
  "sources": [
    {
      "citation_id": 1,
      "video_id": "<hash video ID from context>",
      "url": "<YouTube URL if available, else empty string>",
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS"
    },
    {
      "citation_id": 2,
      "video_id": "<same or different video ID>",
      "url": "<YouTube URL if available, else empty string>",
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS"
    }
  ]
}'''
    
    response: str = Field(
        ...,
        description=(
            "Markdown-formatted response to the user query with inline citations. "
            "Citations are numbers in square brackets (e.g., [1], [2]) that reference "
            "entries in the sources list. Each citation corresponds to ONE timestamp range."
        )
    )
    
    answer_found: bool = Field(
        ...,
        description="Indicates whether the provided context fully answers the user query"
    )
    
    sources: List[CitationSource] = Field(
        default_factory=list,
        description=(
            "List of citation sources. Each entry maps a citation_id to exactly one "
            "timestamp range from a video. Multiple citations can reference the same video "
            "with different timestamp ranges."
        )
    )
