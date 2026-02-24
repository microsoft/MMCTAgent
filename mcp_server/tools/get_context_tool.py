from mmct.video_pipeline.core.tools.get_context import GetContextTool
from mcp_server.config import get_video_agent_provider
from typing import Annotated, Optional, List, Dict, Any
from mcp_server.server import mcp
from loguru import logger

try:
    logger.info("Instantiating GetContextTool")
    _provider = get_video_agent_provider()
    _get_context_tool = GetContextTool(
        embed_provider=_provider.embedding_provider,
        vectordb_chapter=_provider.vectordb_chapter,
    )
    logger.info("Successfully instantiated GetContextTool")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating GetContextTool: {e}")


@mcp.tool(name="get_context_tool")
async def get_context_tool(
    query: Annotated[str, "query for which documents needs to fetch"],
    video_id: Annotated[str, "video id if provided in the instruction"] = None,
    url: Annotated[str, "url if provided in the instruction"] = None,
    fields_to_retrieve: Annotated[
        Optional[List[str]], "list of fields to retrieve from the chapter index"
    ] = None,
    start_time: Annotated[Optional[float], "start time in seconds to filter documents"] = None,
    end_time: Annotated[Optional[float], "end time in seconds to filter documents"] = None,
    top: Annotated[Optional[int], "number of top results to retrieve"] = 3,
) -> List[Dict[str, Any]]:
    if fields_to_retrieve is None:
        fields_to_retrieve = [
            "chapter_transcript",
            "detailed_summary",
            "action_taken",
            "text_from_scene",
            "start_time",
            "end_time",
            "hash_video_id",
            "url",
        ]
    return await _get_context_tool.get_context(
        query=query,
        fields_to_retrieve=fields_to_retrieve,
        video_id=video_id,
        url=url,
        start_time=start_time,
        end_time=end_time,
        top=top,
    )
