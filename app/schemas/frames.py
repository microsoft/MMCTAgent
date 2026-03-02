"""Pydantic schemas for the frame lookup API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class FrameHit(BaseModel):
    """A single frame with its image data."""
    timestamp_second: int = Field(..., description="Timestamp in seconds")
    image_base64: str = Field(..., description="Base64-encoded JPEG image bytes")
    content_type: str = Field(default="image/jpeg", description="MIME type of the image")


class FrameLookupResponse(BaseModel):
    """Response for frame lookup — single frame at the requested timestamp."""
    video_id: str = Field(..., description="Original (pre-normalization) video ID")
    requested_timestamp: int = Field(..., description="Requested timestamp")
    frames: List[FrameHit] = Field(default_factory=list, description="Frame image data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "video_id": "Dk1toyI7AJs",
                "requested_timestamp": 120,
                "frames": [
                    {
                        "timestamp_second": 120,
                        "image_base64": "<base64-encoded JPEG bytes>",
                        "content_type": "image/jpeg"
                    }
                ]
            }
        }
    }
