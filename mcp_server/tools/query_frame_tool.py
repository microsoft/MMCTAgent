from typing import Annotated, Optional
from mmct.video_pipeline.core.tools.query_frame import QueryFrameTool
from mcp_server.config import get_video_agent_provider
from mcp_server.server import mcp
from loguru import logger

try:
    logger.info("Instantiating QueryFrameTool")
    _provider = get_video_agent_provider()
    _query_frame_tool = QueryFrameTool(
        llm_provider=_provider.llm_provider,
        storage_provider=_provider.storage_provider,
        vectordb_keyframes=_provider.vectordb_keyframes,
        image_embedding_provider=_provider.image_embedding_provider,
    )
    logger.info("Successfully instantiated QueryFrameTool")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating QueryFrameTool: {e}")


@mcp.tool(name="query_frame_tool")
async def query_frame_tool(
    query: Annotated[str, "Natural language question about video content to analyze"],
    frame_ids: Annotated[
        Optional[list],
        "List of specific frame filenames to analyze (e.g., ['video_123.jpg', 'video_456.jpg'])",
    ] = None,
    video_id: Annotated[Optional[str], "Unique video identifier hash for frame retrieval"] = None,
    start_time: Annotated[Optional[float], "start time in seconds"] = None,
    end_time: Annotated[Optional[float], "end time in seconds"] = None,
) -> str:
    return await _query_frame_tool.query_frame(
        query=query,
        frame_ids=frame_ids,
        video_id=video_id,
        start_time=start_time,
        end_time=end_time,
    )
