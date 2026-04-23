"""Data retrieval routes for the MMCT MCP Server.

Provides HTTP GET endpoints for fetching video transcripts, frames,
and graph-based video discovery.  These routes are only loaded when the
``ENABLE_DATA_APIS`` environment variable is set to ``true``.

Blob path conventions (matching the ingestion custom steps):
    Transcripts : <normalized-video-id>/transcript.srt   in ``video-transcript-lively``
    Frames      : <normalized-video-id>/<second>/frame.jpg in ``video-frames-lively``
"""

import asyncio
import base64
from typing import Dict

from loguru import logger
from starlette.responses import JSONResponse, PlainTextResponse

from mcp_server.server import mcp, register_api_endpoint
from config.provider_config import get_storage_provider, get_neo4j_query_provider
from mmct.providers.base.database_context import database_override
from mmct.utils.blob_helpers import normalize_video_id

TRANSCRIPT_CONTAINER = "video-transcript-lively"
FRAMES_CONTAINER = "video-frames-lively"

# Register data endpoints on the /api reference page
register_api_endpoint(
    "GET", "/lively/transcript/{video_id}",
    "Get video transcript",
    "Returns the SRT transcript for the given video ID as plain text.",
    tag="Lively",
)
register_api_endpoint(
    "GET", "/lively/frames/{video_id}?ts={int}",
    "Get video frames around a timestamp",
    "Returns base64-encoded JPEG frames at ts-1, ts, and ts+1 (seconds).",
    tag="Lively",
)
register_api_endpoint(
    "GET", "/lively/videos/{database}",
    "List video IDs in a graph database",
    "Returns all unique video IDs from ChapterGroup nodes in the specified Neo4j database.",
    tag="Lively",
)


def _is_not_found(exc: Exception) -> bool:
    """Heuristic check for blob-not-found vs other provider errors."""
    msg = str(exc).lower()
    return "not found" in msg or "404" in msg or "blobnotfound" in msg


# ---------------------------------------------------------------------------
# GET /data/transcript/{video_id}
# ---------------------------------------------------------------------------

@mcp.custom_route("/lively/transcript/{video_id}", methods=["GET"])
async def get_transcript(request):
    """Return the SRT transcript for *video_id* as plain text."""
    video_id: str = request.path_params["video_id"]
    norm_id = normalize_video_id(video_id)
    blob_name = f"{norm_id}/transcript.srt"

    try:
        storage = get_storage_provider()
        data = await storage.load_file_to_memory(TRANSCRIPT_CONTAINER, blob_name)
        return PlainTextResponse(data.decode("utf-8"))
    except Exception as exc:
        if _is_not_found(exc):
            return JSONResponse(
                {"error": f"Transcript not found for video '{video_id}'"},
                status_code=404,
            )
        logger.exception(f"Failed to fetch transcript for {video_id}: {exc}")
        return JSONResponse(
            {"error": "Internal error while fetching transcript"},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# GET /data/frames/{video_id}?ts=<int>
# ---------------------------------------------------------------------------

@mcp.custom_route("/lively/frames/{video_id}", methods=["GET"])
async def get_frames(request):
    """Return base64-encoded JPEG frames at *ts-1*, *ts*, and *ts+1*."""
    video_id: str = request.path_params["video_id"]

    # --- Validate ts query param ---
    ts_raw = request.query_params.get("ts")
    if ts_raw is None:
        return JSONResponse(
            {"error": "Missing required query parameter 'ts' (integer seconds)"},
            status_code=400,
        )
    try:
        ts = int(ts_raw)
    except (ValueError, TypeError):
        return JSONResponse(
            {"error": f"Invalid 'ts' value: '{ts_raw}'. Must be a non-negative integer."},
            status_code=400,
        )
    if ts < 0:
        return JSONResponse(
            {"error": f"Invalid 'ts' value: {ts}. Must be >= 0."},
            status_code=400,
        )

    norm_id = normalize_video_id(video_id)
    timestamps = [t for t in (ts - 1, ts, ts + 1) if t >= 0]

    storage = get_storage_provider()

    async def _fetch(second: int) -> tuple[int, str | None]:
        blob_name = f"{norm_id}/{second}/frame.jpg"
        try:
            data = await storage.load_file_to_memory(FRAMES_CONTAINER, blob_name)
            return second, base64.b64encode(data).decode("ascii")
        except Exception as exc:
            if _is_not_found(exc):
                return second, None
            logger.warning(f"Error fetching frame at ts={second} for {video_id}: {exc}")
            return second, None

    results = await asyncio.gather(*[_fetch(t) for t in timestamps])

    frames: Dict[str, str] = {
        str(sec): b64 for sec, b64 in results if b64 is not None
    }

    if not frames:
        return JSONResponse(
            {"error": f"No frames found for video '{video_id}' around ts={ts}"},
            status_code=404,
        )

    return JSONResponse({
        "video_id": video_id,
        "requested_ts": ts,
        "frames": frames,
    })


# ---------------------------------------------------------------------------
# GET /lively/videos/{database}
# ---------------------------------------------------------------------------

@mcp.custom_route("/lively/videos/{database}", methods=["GET"])
async def get_video_ids(request):
    """Return all unique video IDs from ChapterGroup nodes in *database*."""
    db_name: str = request.path_params["database"]

    provider = get_neo4j_query_provider()
    if provider is None:
        return JSONResponse(
            {"error": "Neo4j is not configured on this server"},
            status_code=503,
        )

    query = """
    MATCH (g:ChapterGroup)
    WHERE g.video_id IS NOT NULL
    RETURN DISTINCT g.video_id AS video_id
    ORDER BY g.video_id
    """

    try:
        async with database_override(db_name):
            records = await provider._run_read(query)
        video_ids = [r["video_id"] for r in records]
    except Exception as exc:
        logger.exception(f"Failed to list video IDs from database '{db_name}': {exc}")
        return JSONResponse(
            {"error": f"Failed to query database '{db_name}': {exc}"},
            status_code=500,
        )

    return JSONResponse({
        "database": db_name,
        "count": len(video_ids),
        "video_ids": video_ids,
    })
