from mmct.video_pipeline.core.tools.get_context import get_context
from typing import Annotated
from mcp_server.server import mcp

@mcp.tool(name="get_context_tool")
async def get_context_tool(
    query: Annotated[str, "query for which documents needs to fetch"],
    index_name: Annotated[str, "vector index name"],
    video_id: Annotated[str, "video id if provided in the instruction"] = None,
    url: Annotated[str, "url if provided in the instruction"] = None,
) -> str:
    return await get_context(
        query=query,
        index_name=index_name,
        video_id=video_id,
        url=url,
    )