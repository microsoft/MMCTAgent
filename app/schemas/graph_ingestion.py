from pydantic import BaseModel, Field
from mmct.video_pipeline import Languages
from typing import Optional


class GraphIngestionRequest(BaseModel):
    """Request schema for graph-based video ingestion operations."""
    language: Languages = Field(..., description="Source language of the video")
    transcript_path: Optional[str] = Field(None, description="Optional path to existing transcript file")
    url: Optional[str] = Field(None, description="Optional URL of the video source")
    max_events_per_chapter: int = Field(10, description="Maximum events to extract per chapter")
    max_objects_per_chapter: int = Field(15, description="Maximum objects to extract per chapter")
    enable_deduplication: bool = Field(True, description="Enable cross-chapter object deduplication")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Graph ingest video with language",
                    "description": "Ingest a video with temporal graph extraction",
                    "value": {
                        "language": "ENGLISH_UNITED_STATES",
                        "transcript_path": None,
                        "url": "https://www.youtube.com/watch?v=example",
                        "max_events_per_chapter": 10,
                        "max_objects_per_chapter": 15,
                        "enable_deduplication": True
                    }
                }
            ]
        }
    }
