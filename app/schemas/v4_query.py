"""V4 Query Request/Response Schemas.

Pydantic models for V4 query API endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class V4QueryRequest(BaseModel):
    """Request schema for V4 Neo4j-backed query operations."""
    
    query: str = Field(
        ..., 
        min_length=1, 
        description="Natural language query about video content"
    )
    video_id: Optional[str] = Field(
        None, 
        description="Optional video ID to scope search to a specific video"
    )
    video_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of video IDs to scope search (for multi-video queries)"
    )
    use_critic: bool = Field(
        default=True, 
        description="Enable critic agent for answer validation"
    )
    stream: bool = Field(
        default=False, 
        description="Enable streaming response (SSE)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Single video query",
                    "description": "Query a specific video by ID",
                    "value": {
                        "query": "How does the farmer collect soil samples?",
                        "video_id": "Dk1toyI7AJs",
                        "use_critic": True,
                        "stream": False
                    }
                },
                {
                    "summary": "Cross-video query",
                    "description": "Search across all videos",
                    "value": {
                        "query": "Which videos show farming techniques?",
                        "use_critic": True,
                        "stream": False
                    }
                },
                {
                    "summary": "Multi-video query",
                    "description": "Search specific videos",
                    "value": {
                        "query": "Compare soil preparation methods",
                        "video_ids": ["Dk1toyI7AJs", "E9sM2b3uV3c"],
                        "use_critic": False,
                        "stream": False
                    }
                }
            ]
        }
    }


class CitationSourceResponse(BaseModel):
    """Citation source in the response."""
    
    citation: str = Field(..., description="Citation marker, e.g., '[1]'")
    video_id: str = Field(..., description="Video identifier")
    start_time: float = Field(..., description="Start timestamp in seconds")
    end_time: float = Field(..., description="End timestamp in seconds")


class V4QueryResponse(BaseModel):
    """Response schema for V4 query operations."""
    
    answer: str = Field(
        ...,
        description="Markdown-formatted answer with inline citations [1], [2], etc."
    )
    sources: List[CitationSourceResponse] = Field(
        default_factory=list,
        description="List of citation sources with video_id and timestamps"
    )
    token_usage: Optional[dict] = Field(
        None,
        description="Token usage statistics (prompt_tokens, completion_tokens)"
    )
    elapsed_seconds: Optional[float] = Field(
        None,
        description="Query processing time in seconds"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The farmer demonstrates soil sampling using the quartering method [1]. He digs a V-shaped cut approximately 15cm deep [2].",
                "sources": [
                    {
                        "citation": "[1]",
                        "video_id": "Dk1toyI7AJs",
                        "start_time": 107.0,
                        "end_time": 163.0
                    },
                    {
                        "citation": "[2]",
                        "video_id": "Dk1toyI7AJs",
                        "start_time": 121.4,
                        "end_time": 127.4
                    }
                ],
                "token_usage": {
                    "prompt_tokens": 1234,
                    "completion_tokens": 567
                },
                "elapsed_seconds": 4.5
            }
        }
    }
