"""V4 Query Service - Process queries using Neo4j graph backend.

Service layer for V4 query API endpoints.
Uses Neo4jQueryProvider with FastEmbedBGEsmallEmbeddingProvider for embeddings.
"""

import json
import os
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional

from loguru import logger
from fastapi import HTTPException

from mmct.v4.orchestrator import V4Orchestrator, process_query_v4
from mmct.v4.query.neo4j_provider import Neo4jQueryProvider
from mmct.v4.schemas import V4QueryResponse
from app.config import get_settings, get_credentials


# Singleton instances
_neo4j_provider: Optional[Neo4jQueryProvider] = None
_embedding_provider = None
_image_embedding_provider = None


def get_neo4j_provider() -> Neo4jQueryProvider:
    """Get or create Neo4j provider singleton."""
    global _neo4j_provider
    if _neo4j_provider is None:
        settings = get_settings()
        
        if not settings.neo4j_password:
            raise ValueError("Neo4j password not configured. Set NEO4J_PASSWORD env variable.")
        
        _neo4j_provider = Neo4jQueryProvider(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        logger.info(f"Neo4j provider initialized: {settings.neo4j_uri}")
    
    return _neo4j_provider


def get_text_embedding_provider():
    """Get text embedding provider (FastEmbedBGEsmall, 384-dim).
    
    This is the same provider used during graph ingestion.
    """
    global _embedding_provider
    if _embedding_provider is None:
        from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider
        _embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
        logger.info("FastEmbedBGEsmallEmbeddingProvider initialized (384-dim)")
    
    return _embedding_provider


def get_image_embedding_provider():
    """Get image embedding provider (QdrantCLIP, 512-dim).
    
    This is the same provider used during graph ingestion for keyframes.
    """
    global _image_embedding_provider
    if _image_embedding_provider is None:
        from mmct.providers.custom_providers import FastEmbedQdrantCLIPEmbeddingProvider
        _image_embedding_provider = FastEmbedQdrantCLIPEmbeddingProvider()
        logger.info("FastEmbedQdrantCLIPEmbeddingProvider initialized (512-dim)")
    
    return _image_embedding_provider


def get_v4_orchestrator(use_critic: bool = True) -> V4Orchestrator:
    """Create V4 orchestrator with all required providers.
    
    Args:
        use_critic: Whether to enable Critic agent.
        
    Returns:
        Configured V4Orchestrator instance.
    """
    from app.config import get_video_agent_provider, get_image_agent_provider
    
    # Get providers
    neo4j_provider = get_neo4j_provider()
    embedding_provider = get_text_embedding_provider()
    image_embedding_provider = get_image_embedding_provider()
    
    # Get video provider for model client and storage
    video_provider = get_video_agent_provider()
    model_client = video_provider.llm_provider.get_autogen_client()
    storage_provider = video_provider.storage_provider
    
    # Get image provider for ImageAgent LLM
    image_provider = get_image_agent_provider()
    
    return V4Orchestrator(
        model_client=model_client,
        neo4j_provider=neo4j_provider,
        embedding_provider=embedding_provider,
        image_embedding_provider=image_embedding_provider,
        storage_provider=storage_provider,
        image_llm_provider=image_provider.llm_provider,
        use_critic=use_critic,
    )


async def process_v4_query(body: dict) -> Dict[str, Any]:
    """Process a V4 query and return the response.
    
    Args:
        body: Request body with query, video_id, etc.
        
    Returns:
        Response dictionary with answer, sources, token_usage.
    """
    query = body.get("query")
    video_id = body.get("video_id")
    video_ids = body.get("video_ids")
    use_critic = body.get("use_critic", True)
    
    if not query:
        raise HTTPException(400, "Query is required")
    
    logger.info(f"V4 Query: {query}")
    if video_id:
        logger.info(f"Scope: video_id={video_id}")
    elif video_ids:
        logger.info(f"Scope: video_ids={video_ids}")
    else:
        logger.info("Scope: cross-video (all)")
    
    try:
        orchestrator = get_v4_orchestrator(use_critic=use_critic)
        
        result = await orchestrator.query(
            user_query=query,
            video_id=video_id,
            video_ids=video_ids,
        )
        
        return result
        
    except Exception as e:
        logger.error(f"V4 query failed: {e}")
        raise HTTPException(500, f"V4 query processing failed: {str(e)}")


async def process_v4_query_stream(body: dict) -> AsyncGenerator[str, None]:
    """Process a V4 query with streaming output (SSE).
    
    Args:
        body: Request body with query, video_id, etc.
        
    Yields:
        SSE formatted events.
    """
    query = body.get("query")
    video_id = body.get("video_id")
    video_ids = body.get("video_ids")
    use_critic = body.get("use_critic", True)
    save_logs = body.get("save_logs", False)
    
    if not query:
        yield _format_sse("error", {"message": "Query is required"})
        return
    
    # Setup logging (only when save_logs is enabled)
    if save_logs:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(logs_dir, f"v4_query_{timestamp}.json")
        events = []
    
    def emit_and_log(event_type: str, data: dict) -> str:
        if save_logs:
            events.append({"type": event_type, **data})
        return _format_sse(event_type, data)
    
    def save_events():
        if save_logs and events:
            with open(log_file, "w") as f:
                json.dump(events, f, indent=2, default=str)
            logger.info(f"V4 query events saved to {log_file}")
    
    try:
        # Initial connection event
        yield emit_and_log("connected", {
            "message": "V4 stream connected",
            "query": query,
            "video_id": video_id,
            "timestamp": datetime.now().isoformat()
        })
        
        orchestrator = get_v4_orchestrator(use_critic=use_critic)
        
        async for message in orchestrator.query_stream(query, video_id, video_ids):
            msg_type = message.get("type", "message")
            
            if msg_type == "final":
                # Final result
                final_data = message.get("data", {})
                yield emit_and_log("complete", {
                    "message": "Query processing complete",
                    "timestamp": datetime.now().isoformat(),
                    "result": final_data,
                })
            else:
                # Intermediate message - ensure content is serializable
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                
                yield emit_and_log("agent_message", {
                    "agent": message.get("agent", "unknown"),
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                })
        
    except Exception as e:
        import traceback
        logger.error(f"V4 streaming failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        yield emit_and_log("error", {
            "message": f"Query processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        })
    
    finally:
        save_events()


def _format_sse(event_type: str, data: dict) -> str:
    """Format data as Server-Sent Event.
    
    Includes 'type' in both the SSE event field AND the JSON data
    for easier client parsing.
    """
    # Include type in data for clients that parse JSON directly
    data_with_type = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(data_with_type)}\n\n"
