"""V5 Query Request/Response Schemas.

Pydantic models for V5 query API endpoints.
Same shape as V4 for compatibility, with pipeline_version marker.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class V5QueryRequest(BaseModel):
    """Request schema for V5 state-machine-driven query."""

    query: str = Field(..., min_length=1, description="Natural language query about video content")
    video_id: Optional[str] = Field(
        None, description="Optional video ID to scope search to a specific video"
    )
    video_ids: Optional[List[str]] = Field(
        None, description="Optional list of video IDs for multi-video queries"
    )
    use_critic: bool = Field(default=True, description="Enable critic for answer validation")
    save_logs: bool = Field(
        default=False, description="Save event logs to file (streaming only)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How does the farmer collect soil samples?",
                    "video_id": "Dk1toyI7AJs",
                    "use_critic": True,
                },
                {"query": "Which videos show farming techniques?", "use_critic": True},
                {
                    "query": "Compare soil preparation methods",
                    "video_ids": ["Dk1toyI7AJs", "E9sM2b3uV3c"],
                    "use_critic": False,
                },
            ]
        }
    }


class CitationSourceResponse(BaseModel):
    """Citation source in the response."""

    citation: str = Field(..., description="Citation marker, e.g., '[1]'")
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")


class V5QueryResponse(BaseModel):
    """Response schema for V5 query operations."""

    answer: str = Field(
        ..., description="Markdown-formatted answer with inline citations [1], [2], etc."
    )
    sources: List[CitationSourceResponse] = Field(
        default_factory=list, description="Citation sources with video_id and timestamps"
    )
    token_usage: Optional[dict] = Field(
        None, description="Token usage statistics"
    )
    elapsed_seconds: Optional[float] = Field(None, description="Processing time in seconds")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    pipeline_version: str = Field(default="v5", description="Pipeline version identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The farmer demonstrates soil sampling using the quartering method [1].",
                "sources": [
                    {
                        "citation": "[1]",
                        "video_id": "Dk1toyI7AJs",
                        "start_time": 107.0,
                        "end_time": 163.0,
                    },
                ],
                "token_usage": {"prompt_tokens": 800, "completion_tokens": 200},
                "elapsed_seconds": 3.2,
                "pipeline_version": "v5",
            }
        }
    }
