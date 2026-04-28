"""Service layer for video query and video query stream endpoints."""

import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from loguru import logger

from config.provider_config import get_acl_callback
from mmct.providers.base.database_context import database_override
from mmct.utils.error_handler import ConfigurationException
from mmct.video_pipeline.query_pipeline import QueryPipelineMode, VideoQueryPipeline


async def run_video_query(
    query: str,
    mode: QueryPipelineMode,
    video_id: Optional[str] = None,
    database: Optional[str] = None,
    user_identifier_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a one-shot video query against the stored knowledge graph.

    Initialises VideoQueryPipeline with use_provider_defaults=True so all
    providers are resolved automatically from environment variables.

    Args:
        query: Natural language question about the video content.
        mode: Pipeline execution mode (graph_agent or graph_state).
        video_id: Optional single video ID scope.
        database: Optional Neo4j database name override.
        user_identifier_context: Per-request caller identity dict; required
            when ACL_ENABLED=true on the server. Forwarded to the pipeline
            unchanged.
    """
    pipeline: Optional[VideoQueryPipeline] = None
    try:
        pipeline = VideoQueryPipeline(
            mode=mode,
            use_provider_defaults=True,
            acl_callback=get_acl_callback(),
        )
        async with database_override(database):
            result = await pipeline.query(
                user_query=query,
                video_id=video_id,
                request_id=str(uuid.uuid4()),
                user_identifier_context=user_identifier_context,
            )
        return result

    except ConfigurationException:
        # Don't echo the exception — its message can be safely surfaced to
        # operators via logs, but interpolating it into an HTTP response
        # risks leaking environment details. Caller correlates via logs.
        logger.exception("Video query pipeline configuration error")
        raise HTTPException(status_code=500, detail="Provider configuration error")
    except Exception:
        logger.exception("Video query pipeline failed")
        raise HTTPException(status_code=500, detail="Video query failed")
    finally:
        if pipeline is not None:
            await pipeline.close()


async def stream_video_query(
    query: str,
    mode: QueryPipelineMode,
    video_id: Optional[str] = None,
    database: Optional[str] = None,
    user_identifier_context: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted JSON strings from VideoQueryPipeline.query_stream().

    Each yielded value is a JSON-serialised event dict from the orchestrator.
    On error, yields a final {"type": "error", "detail": "..."} event before stopping.
    The pipeline is closed in the finally block regardless of outcome.

    Args:
        query: Natural language question about the video content.
        mode: Pipeline execution mode (graph_agent or graph_state).
        video_id: Optional single video ID scope.
        database: Optional Neo4j database name override.
        user_identifier_context: Per-request caller identity dict; required
            when ACL_ENABLED=true on the server. Forwarded to the pipeline
            unchanged.
    """
    pipeline: Optional[VideoQueryPipeline] = None
    try:
        pipeline = VideoQueryPipeline(
            mode=mode,
            use_provider_defaults=True,
            acl_callback=get_acl_callback(),
        )
        async with database_override(database):
            async for event in pipeline.query_stream(
                user_query=query,
                video_id=video_id,
                request_id=str(uuid.uuid4()),
                user_identifier_context=user_identifier_context,
            ):
                yield json.dumps(event)

    except ConfigurationException:
        logger.exception("Video query stream configuration error")
        yield json.dumps({"type": "error", "detail": "Provider configuration error"})
    except Exception:
        logger.exception("Video query stream pipeline failed")
        yield json.dumps({"type": "error", "detail": "Streaming failed"})
    finally:
        if pipeline is not None:
            await pipeline.close()
