from typing import Annotated, Optional, List, Dict, Any
from mmct.video_pipeline.core.tools.get_video_summary import GetVideoSummaryTool
from mcp_server.config import get_video_agent_provider
from mcp_server.server import mcp
from loguru import logger

try:
    logger.info("Instantiating GetVideoSummaryTool")
    _provider = get_video_agent_provider()
    _get_video_summary_tool = GetVideoSummaryTool(
        vectordb_object_registry=_provider.vectordb_object_registry,
        embed_provider=_provider.embedding_provider,
    )
    logger.info("Successfully instantiated GetVideoSummaryTool")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating GetVideoSummaryTool: {e}")


@mcp.tool(name="get_video_summary_tool")
async def get_video_summary_tool(
    query: Annotated[str, "query to search for related video summaries"],
    video_id: Annotated[Optional[str], "unique identifier for the video aka hash Id"] = None,
    url: Annotated[Optional[str], "url of the video"] = None,
    top: Annotated[Optional[int], "number of top results to retrieve (max 3)"] = 3,
) -> List[Dict[str, Any]]:
    return await _get_video_summary_tool.get_video_summary(
        query=query,
        video_id=video_id,
        url=url,
        top=top,
    )
