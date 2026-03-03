from typing_extensions import Annotated, List, Any
from mmct.video_pipeline.core.tools.get_relevant_frames import GetRelevantFrames
from mcp_server.config import get_video_agent_provider
from mcp_server.server import mcp
from loguru import logger

try:
    logger.info("Instantiating GetRelevantFrames")
    _provider = get_video_agent_provider()
    _get_relevant_frames = GetRelevantFrames(
        vectordb_keyframes=_provider.vectordb_keyframes,
        image_embedding_provider=_provider.image_embedding_provider,
    )
    logger.info("Successfully instantiated GetRelevantFrames")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating GetRelevantFrames: {e}")


@mcp.tool(name="get_relevant_frames_tool")
async def get_relevant_frames_tool(
    query: Annotated[str, "query to be look for frames"],
    video_id: Annotated[str, "video id"],
    # index_name: Annotated[str, "search index name"],
    top_k: Annotated[int, "number of relevant frames to fetch"] = 10,
) -> Any:
    return await _get_relevant_frames.get_relevant_frames(
        query=query, 
        video_id=video_id, 
        # index_name=index_name, 
        top_k=top_k
    )
