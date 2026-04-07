
from typing_extensions import Annotated, List
from mmct.video_pipeline.core.tools.get_relevant_frames import get_relevant_frames
from mcp_server.server import mcp

@mcp.tool(name="get_relevant_frames_tool")
async def get_relevant_frames_tool(
    query: Annotated[str, 'query to be look for frames'], 
    video_id: Annotated[str, 'video id'],
    top_k: Annotated[int, 'number of relevant frames to fetch'] = 10
) -> List[str]:
    return get_relevant_frames(
        query=query,
        video_id=video_id,
        top_k=top_k
    )