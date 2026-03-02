"""V4 Response Schema for structured output with citations.

Defines the response format for the V4 query pipeline with:
- Markdown answer with inline citations [1], [2], etc.
- Sources array with video_id and timestamps for playback
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict


class CitationSource(BaseModel):
    """Source information for a single citation reference.
    
    Each citation maps to a specific piece of evidence from the knowledge graph,
    with video_id and timestamp information for playback.
    
    IMPORTANT: start_time and end_time are REQUIRED - never use null.
    """
    citation: str = Field(..., description="The citation marker, e.g., '[1]'")
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., description="Start timestamp in seconds (REQUIRED, never null)")
    end_time: float = Field(..., description="End timestamp in seconds (REQUIRED, never null)")


class V4QueryResponse(BaseModel):
    """Response format for V4 query pipeline.
    
    The answer contains markdown text with inline citation markers [1], [2], etc.
    Each citation corresponds to an entry in the sources array.
    """
    model_config = ConfigDict(extra="forbid")
    
    answer: str = Field(
        ...,
        description=(
            "Markdown-formatted answer with inline citations [1], [2], etc. "
            "Citations reference entries in the sources array. "
            "The answer should be complete and self-contained."
        )
    )
    
    sources: List[CitationSource] = Field(
        default_factory=list,
        description="List of citation sources with video_id and timestamp information"
    )
    
    @staticmethod
    def get_schema_template() -> str:
        """Returns the JSON schema template for use in prompts."""
        return '''{
  "answer": "<Readable answer text with [1], [2] citations. Must be self-contained. Do NOT include sources list, timestamps, video IDs, or graph terms (ChapterGroup, Chapter, etc.) here — only the human-readable answer.>",
  "sources": [
    {
      "citation": "[1]",
      "video_id": "<video_id>",
      "start_time": <number - REQUIRED, never null>,
      "end_time": <number - REQUIRED, never null>
    }
  ]
}

CRITICAL: start_time and end_time MUST be numbers (float/int). NEVER use null.
If no specific timestamp, use 0.0 for start_time and video duration for end_time.
Each citation marker [1], [2], etc. should appear ONLY ONCE in sources array.
The "answer" field must contain ONLY the answer text — NO source lists, NO timestamps, NO video IDs, NO graph node types.'''

    @staticmethod
    def get_example() -> str:
        """Returns an example response for prompts."""
        return '''{
  "answer": "The presenter demonstrates the installation process using the bracket method [1]. He positions the component at a 45-degree angle and secures it with four screws [2]. The final assembly is tested by applying pressure to verify stability [3].",
  "sources": [
    {"citation": "[1]", "video_id": "abc123", "start_time": 107.0, "end_time": 163.0},
    {"citation": "[2]", "video_id": "abc123", "start_time": 121.4, "end_time": 127.4},
    {"citation": "[3]", "video_id": "abc123", "start_time": 341.0, "end_time": 393.0}
  ]
}'''
