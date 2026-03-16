from pydantic import BaseModel, Field
from mmct.video_pipeline import Languages
from typing import Optional

class IngestionRequest(BaseModel):
    """Request schema for video ingestion operations."""
    language: Languages = Field(..., description="Source language of the video")
    transcript_path: Optional[str] = Field(None, description="Optional path to existing transcript file")
    url: Optional[str] = Field(None, description="Optional URL of the video source")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Ingest video with language",
                    "description": "Ingest a video with English language",
                    "value": {
                        "language": "ENGLISH_UNITED_STATES",
                        "transcript_path": None,
                        "url": "https://www.youtube.com/watch?v=example"
                    }
                }
            ]
        }
    }