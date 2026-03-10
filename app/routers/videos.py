"""Videos router - API endpoints for video metadata from Neo4j graph."""

import logging

from fastapi import APIRouter, HTTPException
from app.services.v4_query_service import get_neo4j_provider
from app.services.video_catalog_service import DEFAULT_CATALOG_MAX_TOKENS

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


@router.get(
    "/videos/catalog",
    summary="Get Video Catalog",
    description="Returns the cached video catalog string used in the Planner Agent's system prompt.",
)
async def get_video_catalog():
    """Return the cached video catalog."""
    from app.services.video_catalog_service import get_cached_catalog

    catalog = get_cached_catalog()
    if catalog is None:
        raise HTTPException(status_code=503, detail="Video catalog not yet generated. Server may still be starting up.")
    return {"catalog": catalog, "length_chars": len(catalog)}


@router.post(
    "/videos/catalog/refresh",
    summary="Refresh Video Catalog",
    description="Re-fetches video metadata from Neo4j and regenerates the LLM-compressed catalog.",
)
async def refresh_video_catalog(max_tokens: int = DEFAULT_CATALOG_MAX_TOKENS):
    """Regenerate the video catalog from Neo4j and update the cache."""
    from app.services.video_catalog_service import generate_video_catalog
    from app.config import get_video_agent_provider

    try:
        neo4j_provider = get_neo4j_provider()
        llm_provider = get_video_agent_provider().llm_provider
        catalog = await generate_video_catalog(neo4j_provider, llm_provider, max_tokens=max_tokens)
        return {"catalog": catalog, "length_chars": len(catalog), "max_tokens": max_tokens, "refreshed": True}
    except Exception as e:
        logger.error(f"Video catalog refresh failed: {e}")
        raise HTTPException(status_code=500, detail=f"Catalog refresh failed: {e}")
