"""V5 Response Schema for structured output with citations.

Defines the response format for the V5 query pipeline.
Same structure as V4 for API compatibility.
"""

from typing import List
from pydantic import BaseModel, Field, ConfigDict


class CitationSource(BaseModel):
    """Source information for a single citation reference."""

    citation: str = Field(..., description="The citation marker, e.g., '[1]'")
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(
        ..., description="Start timestamp in seconds (REQUIRED, never null)"
    )
    end_time: float = Field(
        ..., description="End timestamp in seconds (REQUIRED, never null)"
    )


class V5QueryResponse(BaseModel):
    """Response format for V5 query pipeline."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description=(
            "Markdown-formatted answer with inline citations [1], [2], etc. "
            "Citations reference entries in the sources array. "
            "The answer should be complete and self-contained."
        ),
    )

    sources: List[CitationSource] = Field(
        default_factory=list,
        description="List of citation sources with video_id and timestamp information",
    )

    @staticmethod
    def get_schema_template() -> str:
        return """{
  "answer": "<Readable answer text with [1], [2] citations. Must be self-contained. No source lists, timestamps, video IDs, or graph terms here.>",
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
Each citation marker [1], [2], etc. should appear ONLY ONCE in sources array."""
