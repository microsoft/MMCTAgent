"""
Video Query Tool for MMCT MCP Server.

This module provides a tool for querying ingested video content using specialized
MMCT pipelines, including agentic graph traversal and deterministic state machines.
"""

from typing import Optional, List, Literal
from loguru import logger
from mcp_server.server import mcp
from mmct.video_pipeline.query_pipeline import VideoQueryPipeline, QueryPipelineMode
from config.provider_config import get_query_pipeline_providers

# Singleton pipeline instances — one per mode, reused across requests.
_pipelines: dict[str, VideoQueryPipeline] = {}


def get_pipeline(mode: str) -> VideoQueryPipeline:
    """Return a singleton VideoQueryPipeline for the given mode."""
    if mode not in _pipelines:
        _pipelines[mode] = VideoQueryPipeline(
            mode=mode,
            use_provider_defaults=False,
            **get_query_pipeline_providers().__dict__
        )
    return _pipelines[mode]

@mcp.tool(
    name="video_query",
    description="""
    Query ingested video content using either agentic graph traversal or a deterministic state machine.
    
    Args:
        query: Natural language question about the video content.
        video_id: Optional specific video ID to query.
        video_ids: Optional list of video IDs to scope the query.
        mode: Pipeline mode to use. 'graph_agent' for complex agentic reasoning, 
              'graph_state' for deterministic extraction (default: 'graph_state').
    """
)
async def video_query(
    query: str,
    video_id: Optional[str] = None,
    video_ids: Optional[List[str]] = None,
    mode: Literal["graph_agent", "graph_state"] = "graph_state",
) -> dict:
    """
    Executes a query against the video knowledge graph or state machine index.

    Args:
        query (str): The natural language question about the video's content.
        video_id (str, optional): A single video identifier to scope the query. 
            Defaults to None (searches across all ingested videos).
        video_ids (List[str], optional): A list of video identifiers to scope the 
            query across multiple specific videos. Defaults to None.
        mode (str, optional): The pipeline execution strategy. 
            'graph_agent' uses an autonomous swarm of agents; 'graph_state' 
            uses a deterministic state machine. Defaults to 'graph_state'.

    Returns:
        dict: The structured query result containing the answer, sources, and metadata.
    """
    try:
        logger.info(f"Executing video_query: {query} (mode={mode}, video_id={video_id}, video_ids={video_ids})")
        
        pipeline = get_pipeline(mode)
        
        # Execute the query through the pipeline orchestrator
        result = await pipeline.query(
            user_query=query,
            video_id=video_id,
            video_ids=video_ids
        )
        
        return result
        
    except Exception as e:
        logger.exception(f"Exception generated during video_query execution: {e}")
        return {"error": str(e), "status": "failed"}
