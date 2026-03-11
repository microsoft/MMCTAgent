"""V5 Query Service — Process queries using state machine pipeline.

Service layer for V5 query API endpoints.
Reuses existing provider singletons from v4 service.
"""

import json
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional

from loguru import logger
from fastapi import HTTPException

from mmct.v5.orchestrator import V5Orchestrator
from mmct.v5.query.neo4j_provider import Neo4jQueryProvider
from app.config import get_settings, get_credentials


# Reuse V4's singleton providers (same Neo4j, same embeddings)
def _get_neo4j_provider() -> Neo4jQueryProvider:
    """Get or create Neo4j provider singleton (shared with V4)."""
    # Import V4's singleton to reuse the same connection
    from app.services.v4_query_service import get_neo4j_provider as _v4_neo4j
    return _v4_neo4j()


def _get_text_embedding_provider():
    from app.services.v4_query_service import get_text_embedding_provider as _v4_emb
    return _v4_emb()


def _get_image_embedding_provider():
    from app.services.v4_query_service import get_image_embedding_provider as _v4_img_emb
    return _v4_img_emb()


def get_v5_orchestrator(use_critic: bool = True) -> V5Orchestrator:
    """Create V5 orchestrator with all required providers."""
    from app.config import get_video_agent_provider, get_image_agent_provider
    from app.services.video_catalog_service import get_cached_catalog

    neo4j_provider = _get_neo4j_provider()
    embedding_provider = _get_text_embedding_provider()
    image_embedding_provider = _get_image_embedding_provider()

    video_provider = get_video_agent_provider()
    model_client = video_provider.llm_provider.get_autogen_client()
    storage_provider = video_provider.storage_provider

    image_provider = get_image_agent_provider()

    return V5Orchestrator(
        model_client=model_client,
        neo4j_provider=neo4j_provider,
        embedding_provider=embedding_provider,
        image_embedding_provider=image_embedding_provider,
        storage_provider=storage_provider,
        image_llm_provider=image_provider.llm_provider,
        use_critic=use_critic,
        video_catalog=get_cached_catalog(),
    )


async def process_v5_query(body: dict, request_id: str = "") -> Dict[str, Any]:
    """Process a V5 query and return the response."""
    query = body.get("query")
    video_id = body.get("video_id")
    video_ids = body.get("video_ids")
    use_critic = body.get("use_critic", True)

    if not query:
        raise HTTPException(400, "Query is required")

    logger.info(f"[{request_id}] V5 Query: {query}")
    if video_id:
        logger.info(f"[{request_id}] Scope: video_id={video_id}")
    elif video_ids:
        logger.info(f"[{request_id}] Scope: video_ids={video_ids}")
    else:
        logger.info(f"[{request_id}] Scope: cross-video (all)")

    try:
        orchestrator = get_v5_orchestrator(use_critic=use_critic)

        result = await orchestrator.query(
            user_query=query,
            video_id=video_id,
            video_ids=video_ids,
            request_id=request_id,
        )

        result["request_id"] = request_id
        result["pipeline_version"] = "v5"
        return result

    except Exception as e:
        logger.error(f"[{request_id}] V5 query failed: {e}")
        raise HTTPException(500, f"V5 query processing failed: {str(e)}")


async def process_v5_query_stream(body: dict, request_id: str = "") -> AsyncGenerator[str, None]:
    """Process a V5 query with streaming output (SSE)."""
    query = body.get("query")
    video_id = body.get("video_id")
    video_ids = body.get("video_ids")
    use_critic = body.get("use_critic", True)
    save_logs = body.get("save_logs", False)

    if not query:
        yield _format_sse("error", {"message": "Query is required"})
        return

    if save_logs:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(logs_dir, f"v5_query_{timestamp}.json")
        events = []

    def emit_and_log(event_type: str, data: dict) -> str:
        if save_logs:
            events.append({"type": event_type, **data})
        return _format_sse(event_type, data)

    def save_events():
        if save_logs and events:
            with open(log_file, "w") as f:
                json.dump(events, f, indent=2, default=str)
            logger.info(f"[{request_id}] V5 query events saved to {log_file}")

    try:
        yield emit_and_log(
            "connected",
            {
                "message": "V5 stream connected",
                "query": query,
                "video_id": video_id,
                "request_id": request_id,
                "pipeline_version": "v5",
                "timestamp": datetime.now().isoformat(),
            },
        )

        orchestrator = get_v5_orchestrator(use_critic=use_critic)

        async for message in orchestrator.query_stream(
            query, video_id, video_ids, request_id=request_id
        ):
            msg_type = message.get("type", "message")

            if msg_type == "final":
                final_data = message.get("data", {})
                final_data["request_id"] = request_id
                final_data["pipeline_version"] = "v5"
                yield emit_and_log(
                    "complete",
                    {
                        "message": "Query processing complete",
                        "timestamp": datetime.now().isoformat(),
                        "result": final_data,
                    },
                )
            else:
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)

                yield emit_and_log(
                    "agent_message",
                    {
                        "agent": message.get("agent", "unknown"),
                        "content": content,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

    except Exception as e:
        import traceback

        logger.error(f"[{request_id}] V5 streaming failed: {e}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        yield emit_and_log(
            "error",
            {
                "message": f"Query processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            },
        )

    finally:
        save_events()


def _format_sse(event_type: str, data: dict) -> str:
    data_with_type = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(data_with_type, default=str)}\n\n"
