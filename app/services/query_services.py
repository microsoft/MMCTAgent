"""Service layer for video and image query endpoints.

Provides the business logic behind the query API routes. Delegates to the
MMCT agent classes (VideoAgent, ImageAgent) and handles streaming vs
non-streaming response modes, error propagation, and temporary file lifecycle.
"""

import os
import tempfile

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mmct.video_pipeline import VideoAgent

from app.config import get_video_agent_provider, get_image_agent_provider


# ---------------------------------------------------------------------------
# Image query
# ---------------------------------------------------------------------------

async def process_image_query(file: UploadFile, body: dict):
    """Process an image analysis query using the configured ImageAgent.

    Saves the uploaded file to a temporary path, builds an ImageAgent with the
    requested tools and critic settings, and returns either a direct response or
    a streaming NDJSON response.

    Args:
        file: Uploaded image file.
        body: Parsed request body containing 'query', 'tools', 'use_critic_agent',
              and optional 'stream' flag.

    Returns:
        ImageAgent response object, or a StreamingResponse for stream=True.

    Raises:
        HTTPException 400: If an unknown tool name is requested.
        HTTPException 500: If the agent raises an unexpected error.
    """
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        tool_list = [getattr(ImageQnaTools, t) for t in body["tools"]]
    except AttributeError:
        os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Invalid tool name in request")

    provider = get_image_agent_provider()
    stream_enabled = body.get("stream", False)

    agent = ImageAgent(
        query=body["query"],
        image_path=tmp_path,
        tools=tool_list,
        use_critic_agent=body["use_critic_agent"],
        stream=stream_enabled,
        provider=provider,
        use_console=not stream_enabled,
    )

    try:
        response = await agent()
        if stream_enabled:
            return StreamingResponse(
                _stream_generator(response),
                media_type="application/x-ndjson",
            )
        return response
    except Exception as exc:
        logger.error(f"Image query failed: {exc}")
        raise HTTPException(status_code=500, detail="Image processing failed")
    finally:
        if not stream_enabled and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Video query
# ---------------------------------------------------------------------------

async def process_video_query(body: dict):
    """Process a video question-answering query using the V5 Neo4j-backed orchestrator.

    Routes through V5Orchestrator which queries Neo4j directly — matching the
    temporal graph ingestion pipeline used in v1.

    Args:
        body: Parsed request body containing 'query', optional 'video_id', 'url',
              'use_critic_agent', 'stream', and 'cache' fields.

    Returns:
        VideoAgentResponse-compatible dict, or a StreamingResponse for stream=True.

    Raises:
        HTTPException 500: If the agent raises an unexpected error.
    """
    from app.services.neo4j_query_service import process_video_query as process_neo4j_query

    try:
        result = await process_neo4j_query(body)
        answer = result.get("answer", "")
        return {
            "response": answer,
            "answer_found": bool(answer and answer != "No answer generated."),
            "source": result.get("sources", []),
            "tokens": result.get("token_usage"),
        }
    except Exception as exc:
        logger.error(f"Video query failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Video query processing failed: {exc}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _stream_generator(agent_response):
    """Yield serialised JSON lines from an agent async generator.

    Args:
        agent_response: Async generator returned by an agent in stream mode.

    Yields:
        JSON-serialised string per event, newline-terminated.
    """
    import json

    async for chunk in agent_response:
        if isinstance(chunk, dict):
            yield json.dumps(chunk) + "\n"
        else:
            yield json.dumps({"type": chunk.__class__.__name__, "content": str(chunk)}) + "\n"
