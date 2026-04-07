"""Pydantic request / response schemas for MMCT query endpoints."""

from typing import List, Optional

from mmct.image_pipeline import ImageQnaTools
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Image query
# ---------------------------------------------------------------------------

class ImageQueryRequest(BaseModel):
    """Request schema for image analysis queries."""

    query: str = Field(..., min_length=1, description="Natural language question about the image")
    tools: List[str] = Field(
        ...,
        description="Analysis tools to run. Available: object_detection, ocr, recog, vit",
        examples=[["ocr"], ["object_detection", "recog"]],
    )
    use_critic_agent: bool = Field(default=True, description="Enable critic agent for answer validation")
    stream: bool = Field(default=False, description="Stream agent events as NDJSON")

    @field_validator("tools")
    @classmethod
    def parse_tools(cls, v):
        """Accept either a list of strings or a single comma-separated string."""
        if isinstance(v, list) and len(v) == 1:
            return [t.strip() for t in v[0].split(",") if t.strip()]
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What text is visible in this image?",
                    "tools": ["ocr"],
                    "use_critic_agent": False,
                    "stream": False,
                },
                {
                    "query": "Identify and describe all objects",
                    "tools": ["object_detection", "recog"],
                    "use_critic_agent": True,
                    "stream": False,
                },
            ]
        }
    }


# ---------------------------------------------------------------------------
# Video query
# ---------------------------------------------------------------------------

class VideoQueryRequest(BaseModel):
    """Request schema for video question-answering queries."""

    query: str = Field(..., min_length=1, description="Natural language question about video content")
    video_id: Optional[str] = Field(
        None, description="Scope the search to a specific ingested video"
    )
    url: Optional[str] = Field(
        None, description="Optional video URL to filter search results"
    )
    use_critic_agent: bool = Field(default=True, description="Enable critic agent for answer validation")
    stream: bool = Field(default=False, description="Stream agent events as NDJSON")
    cache: bool = Field(default=False, description="Enable response caching")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What topics are covered in this video?",
                    "video_id": "Dk1toyI7AJs",
                    "use_critic_agent": True,
                    "stream": False,
                },
                {
                    "query": "Summarise the key points",
                    "use_critic_agent": True,
                    "stream": True,
                },
            ]
        }
    }


class VideoQueryResponse(BaseModel):
    """Response schema for video question-answering queries.

    Mirrors the structure returned by VideoAgent. The 'source' field contains
    references to the video segments used to construct the answer.
    """

    response: str = Field(..., description="Natural language answer to the query")
    answer_found: bool = Field(..., description="Whether a relevant answer was found in the video")
    source: List[dict] = Field(
        default_factory=list,
        description="Source segments with video_id, blob_url, url, and timestamps",
    )
    tokens: Optional[dict] = Field(None, description="Token usage statistics")
