"""Service layer for video query and video query stream endpoints."""

import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException
from loguru import logger

from mmct.utils.error_handler import ConfigurationException
from mmct.video_pipeline.query_pipeline import QueryPipelineMode, VideoQueryPipeline


async def run_video_query(
    query: str,
    mode: QueryPipelineMode,
    video_id: str,
) -> Dict[str, Any]:
    """
    Execute a one-shot video query against the stored knowledge graph.

    Initialises VideoQueryPipeline with use_provider_defaults=True so all
    providers are resolved automatically from environment variables.
    """
    pipeline: Optional[VideoQueryPipeline] = None
    try:
        pipeline = VideoQueryPipeline(mode=mode, use_provider_defaults=True)
        result = await pipeline.query(
            user_query=query,
            video_id=video_id,
            request_id=str(uuid.uuid4()),
        )
        return result

    except ConfigurationException as exc:
        raise HTTPException(status_code=500, detail=f"Provider configuration error: {exc}")
    except Exception as exc:
        logger.exception("Video query pipeline failed")
        raise HTTPException(status_code=500, detail=f"Video query failed: {exc}")
    finally:
        if pipeline is not None:
            await pipeline.close()


async def stream_video_query(
    query: str,
    mode: QueryPipelineMode,
    video_id: str,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted JSON strings from VideoQueryPipeline.query_stream().

    Each yielded value is a JSON-serialised event dict from the orchestrator.
    On error, yields a final {"type": "error", "detail": "..."} event before stopping.
    The pipeline is closed in the finally block regardless of outcome.
    """
    pipeline: Optional[VideoQueryPipeline] = None
    try:
        pipeline = VideoQueryPipeline(mode=mode, use_provider_defaults=True)
        async for event in pipeline.query_stream(
            user_query=query,
            video_id=video_id,
            request_id=str(uuid.uuid4()),
        ):
            yield json.dumps(event)

    except ConfigurationException as exc:
        yield json.dumps({"type": "error", "detail": f"Provider configuration error: {exc}"})
    except Exception as exc:
        logger.exception("Video query stream pipeline failed")
        yield json.dumps({"type": "error", "detail": f"Streaming failed: {exc}"})
    finally:
        if pipeline is not None:
            await pipeline.close()
