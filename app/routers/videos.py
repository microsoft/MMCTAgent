"""Videos router - API endpoints for video metadata from Neo4j graph."""

import logging

from fastapi import APIRouter, HTTPException
from app.services.v4_query_service import get_neo4j_provider

logger = logging.getLogger(__name__)
router = APIRouter(tags=["videos"])


@router.get(
    "/videos",
    summary="List Ingested Videos",
    description="Returns the list of all video IDs that have been ingested into the Neo4j knowledge graph.",
    response_model=dict,
    responses={
        200: {
            "description": "List of video IDs",
            "content": {
                "application/json": {
                    "example": {
                        "video_ids": ["Dk1toyI7AJs", "0qnRu8GNxBg", "2lp4VeuE6OM"],
                        "count": 3,
                    }
                }
            },
        }
    },
)
async def list_videos():
    """Return all unique video IDs from the Neo4j graph."""
    try:
        provider = get_neo4j_provider()
    except (ValueError, Exception) as e:
        logger.error(f"Neo4j provider initialization failed: {e}")
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")
    video_ids = await provider.get_all_video_ids()
    return {"video_ids": video_ids, "count": len(video_ids)}
