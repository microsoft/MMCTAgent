"""Pydantic schemas for the transcript lookup API."""

from typing import Optional
from pydantic import BaseModel, Field


class TranscriptLookupResponse(BaseModel):
    """Response for transcript lookup."""
    video_id: str = Field(..., description="Original (pre-normalization) video ID")
    blob_url: str = Field(..., description="Blob URL for the SRT transcript file")

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_id": "Dk1toyI7AJs",
                "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-transcript-lively/Dk1toyI7AJs/transcript.srt"
            }
        }
    }
