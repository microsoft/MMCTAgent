"""Schemas for the ingestion endpoint.

LanguageCode is a str-backed enum dynamically derived from the canonical
Languages enum so Swagger UI renders it as a 148-option dropdown and the list
stays automatically in sync if Languages ever gains new members.
"""

from enum import Enum as _Enum
from typing import Optional

from pydantic import BaseModel, Field

from mmct.video_pipeline.core.ingestion.languages import Languages

# Dynamically produce a str-backed enum for Swagger dropdown rendering.
LanguageCode: type = _Enum(  # type: ignore[assignment]
    "LanguageCode",
    {name: member.value for name, member in Languages.__members__.items()},
    type=str,
)
LanguageCode.__doc__ = "BCP-47 language code for video transcription."


class IngestionResponse(BaseModel):
    """Response returned by POST /ingestion."""

    video_id: str = Field(..., description="SHA-256 hash used as the unique video identifier")
    status: str = Field(..., description="Pipeline completion status: 'completed' or 'failed'")
    duration_seconds: Optional[float] = Field(None, description="Total wall-clock time of the pipeline run")
    message: str = Field(default="", description="Human-readable status message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "video_id": "a3f2c1d8e9b74052abc123def456789012345678901234567890abcdef012345",
                    "status": "completed",
                    "duration_seconds": 142.7,
                    "message": "Ingestion pipeline completed successfully.",
                }
            ]
        }
    }
