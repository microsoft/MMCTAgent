"""Schemas for video query endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# QueryPipelineMode is already (str, Enum) — FastAPI renders it as a Swagger dropdown natively.
from mmct.video_pipeline.query_pipeline import QueryPipelineMode  # noqa: F401 — re-exported for routers


class VideoQueryResponse(BaseModel):
    """Response returned by POST /video-query."""

    answer: str = Field(..., description="Natural-language answer to the query")
    sources: List[Any] = Field(default_factory=list, description="Source segments used to build the answer")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="LLM token consumption breakdown")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The presenter demonstrates how to configure the Azure endpoint at the 2:34 mark, then walks through environment variable setup.",
                    "sources": [
                        {"chapter": "Azure Configuration", "timestamp": "02:34", "score": 0.92}
                    ],
                    "token_usage": {"prompt_tokens": 512, "completion_tokens": 128},
                }
            ]
        }
    }
