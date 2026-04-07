"""Neo4j query service — processes video queries using the graph-backed pipeline.

Routes queries through the orchestrator which performs temporal graph traversal,
semantic search, and optional critic validation against the Neo4j knowledge graph.
"""

import json
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any

from loguru import logger
from fastapi import HTTPException

from mmct.v5.orchestrator import V5Orchestrator
from mmct.v5.query.neo4j_provider import Neo4jQueryProvider
from app.config import get_settings


_neo4j_provider = None


def _get_neo4j_provider() -> Neo4jQueryProvider:
    """Get or create Neo4j provider singleton."""
    global _neo4j_provider
    if _neo4j_provider is None:
        settings = get_settings()
        _neo4j_provider = Neo4jQueryProvider(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        logger.info("Neo4j provider initialized")
    return _neo4j_provider


def _get_text_embedding_provider():
    from app.services.embedding_providers_service import get_text_embedding_provider
    return get_text_embedding_provider()


def _get_image_embedding_provider():
    from app.services.embedding_providers_service import get_image_embedding_provider
    return get_image_embedding_provider()


def get_orchestrator(use_critic: bool = True) -> V5Orchestrator:
    """Create orchestrator with all required providers."""
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


async def process_video_query(body: dict, request_id: str = "") -> Dict[str, Any]:
    """Process a video query and return the response.

    Args:
        body: Request body with query, video_id, video_ids, use_critic fields.
        request_id: Unique request identifier for log correlation.

    Returns:
        Response dictionary with answer, sources, token_usage, request_id.

    Raises:
        HTTPException 400: If query is missing.
        HTTPException 500: If the orchestrator raises an unexpected error.
    """
    query = body.get("query")
    video_id = body.get("video_id")
    video_ids = body.get("video_ids")
    use_critic = body.get("use_critic", True)

    if not query:
        raise HTTPException(400, "Query is required")

    logger.info(f"[{request_id}] Query: {query}")
    if video_id:
        logger.info(f"[{request_id}] Scope: video_id={video_id}")
    elif video_ids:
        logger.info(f"[{request_id}] Scope: video_ids={video_ids}")
    else:
        logger.info(f"[{request_id}] Scope: cross-video (all)")

    try:
        orchestrator = get_orchestrator(use_critic=use_critic)
        result = await orchestrator.query(
            user_query=query,
            video_id=video_id,
            video_ids=video_ids,
            request_id=request_id,
        )
        result["request_id"] = request_id
        return result

    except Exception as exc:
        logger.error(f"[{request_id}] Query failed: {exc}")
        raise HTTPException(500, f"Query processing failed: {str(exc)}")


async def process_video_query_stream(body: dict, request_id: str = "") -> AsyncGenerator[str, None]:
    """Process a video query with streaming output (SSE).

    Args:
        body: Request body with query, video_id, video_ids, use_critic, save_logs fields.
        request_id: Unique request identifier for log correlation.

    Yields:
        SSE formatted event strings.
    """
    query = body.get("query")
    video_id = body.get("video_id")
    video_ids = body.get("video_ids")
    use_critic = body.get("use_critic", True)
    save_logs = body.get("save_logs", False)

    if not query:
        yield _format_sse("error", {"message": "Query is required"})
        return

    events = []
    log_file = None
    if save_logs:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(logs_dir, f"query_{timestamp}.json")

    def emit_and_log(event_type: str, data: dict) -> str:
        if save_logs:
            events.append({"type": event_type, **data})
        return _format_sse(event_type, data)

    def save_events():
        if save_logs and events and log_file:
            with open(log_file, "w") as f:
                json.dump(events, f, indent=2, default=str)
            logger.info(f"[{request_id}] Query events saved to {log_file}")

    try:
        yield emit_and_log(
            "connected",
            {
                "message": "Stream connected",
                "query": query,
                "video_id": video_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

        orchestrator = get_orchestrator(use_critic=use_critic)

        async for message in orchestrator.query_stream(
            query, video_id, video_ids, request_id=request_id
        ):
            msg_type = message.get("type", "message")

            if msg_type == "final":
                final_data = message.get("data", {})
                final_data["request_id"] = request_id
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

    except Exception as exc:
        import traceback
        logger.error(f"[{request_id}] Streaming failed: {exc}")
        logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
        yield emit_and_log(
            "error",
            {
                "message": f"Query processing failed: {str(exc)}",
                "timestamp": datetime.now().isoformat(),
            },
        )

    finally:
        save_events()


def _format_sse(event_type: str, data: dict) -> str:
    """Format data as a Server-Sent Event string."""
    data_with_type = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(data_with_type, default=str)}\n\n"
