"""Pydantic schemas for the frame lookup API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class FrameHit(BaseModel):
    """A single frame blob URL with its timestamp."""
    timestamp_second: int = Field(..., description="Timestamp in seconds")
    blob_url: str = Field(..., description="Full blob URL for the frame")


class FrameLookupResponse(BaseModel):
    """Response for frame lookup — up to 3 URLs for ±1 second window."""
    video_id: str = Field(..., description="Original (pre-normalization) video ID")
    requested_timestamp: int = Field(..., description="Requested centre timestamp")
    frames: List[FrameHit] = Field(default_factory=list, description="Existing frame URLs")

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_id": "Dk1toyI7AJs",
                "requested_timestamp": 120,
                "frames": [
                    {
                        "timestamp_second": 119,
                        "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-frames-lively/Dk1toyI7AJs/119/frame.jpg"
                    },
                    {
                        "timestamp_second": 120,
                        "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-frames-lively/Dk1toyI7AJs/120/frame.jpg"
                    },
                    {
                        "timestamp_second": 121,
                        "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-frames-lively/Dk1toyI7AJs/121/frame.jpg"
                    }
                ]
            }
        }
    }
