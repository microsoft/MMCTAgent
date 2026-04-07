"""Videos router — endpoints for querying the video catalog and metadata."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.config import get_neo4j_query_provider
from app.services.video_catalog_service import DEFAULT_CATALOG_MAX_TOKENS

router = APIRouter(tags=["videos"])


@router.get(
    "/videos",
    summary="List Ingested Videos",
    description="Returns all video IDs that have been ingested into the knowledge graph.",
    response_model=dict,
    responses={
        200: {
            "description": "List of video IDs",
            "content": {
                "application/json": {
                    "example": {
                        "video_ids": ["Dk1toyI7AJs", "0qnRu8GNxBg"],
                        "count": 2,
                    }
                }
            },
        },
        503: {"description": "Graph store unavailable"},
    },
)
async def list_videos():
    """Return all unique video IDs present in the knowledge graph."""
    try:
        provider = get_neo4j_query_provider()
    except (ValueError, Exception) as exc:
        logger.error(f"Neo4j provider initialisation failed: {exc}")
        raise HTTPException(status_code=503, detail=f"Graph store unavailable: {exc}")

    try:
        video_ids = await provider.get_all_video_ids()
    except Exception as exc:
        logger.error(f"Failed to fetch video IDs: {exc}")
        raise HTTPException(status_code=500, detail="Failed to retrieve video list")

    return {"video_ids": video_ids, "count": len(video_ids)}


@router.get(
    "/videos/catalog",
    summary="Get Video Catalog",
    description=(
        "Returns the cached video catalog string. "
        "The catalog is generated at startup and used by the query agent to understand "
        "what content is available before formulating search queries."
    ),
    responses={
        503: {"description": "Catalog not yet generated"},
    },
)
async def get_video_catalog():
    """Return the cached video catalog, or 503 if not yet generated."""
    from app.services.video_catalog_service import get_cached_catalog

    catalog = get_cached_catalog()
    if catalog is None:
        raise HTTPException(
            status_code=503,
            detail="Video catalog not yet generated. Server may still be starting up.",
        )
    return {"catalog": catalog, "length_chars": len(catalog)}


@router.post(
    "/videos/catalog/refresh",
    summary="Refresh Video Catalog",
    description=(
        "Re-fetches video metadata from the knowledge graph and regenerates the "
        "LLM-compressed catalog string. Useful after ingesting new videos."
    ),
)
async def refresh_video_catalog(max_tokens: int = DEFAULT_CATALOG_MAX_TOKENS):
    """Regenerate the video catalog and update the in-memory cache."""
    from app.services.video_catalog_service import generate_video_catalog
    from app.config import get_video_agent_provider

    try:
        neo4j_provider = get_neo4j_query_provider()
        llm_provider = get_video_agent_provider().llm_provider
        catalog = await generate_video_catalog(neo4j_provider, llm_provider, max_tokens=max_tokens)
        return {
            "catalog": catalog,
            "length_chars": len(catalog),
            "max_tokens": max_tokens,
            "refreshed": True,
        }
    except Exception as exc:
        logger.error(f"Video catalog refresh failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Catalog refresh failed: {exc}")
