"""Pydantic schemas for the transcript lookup API."""

from typing import Optional
from pydantic import BaseModel, Field


class TranscriptLookupResponse(BaseModel):
    """Response for transcript lookup — returns the full transcript content."""
    video_id: str = Field(..., description="Original (pre-normalization) video ID")
    transcript: str = Field(..., description="Full transcript content (SRT format)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_id": "Dk1toyI7AJs",
                "transcript": "1\n00:00:00,000 --> 00:00:05,000\nHello and welcome to this lecture.\n\n2\n00:00:05,000 --> 00:00:10,000\nToday we will discuss..."
            }
        }
    }
