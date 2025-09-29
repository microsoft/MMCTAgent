
from datetime import time
from typing import Annotated, Optional
from mmct.video_pipeline.core.tools.query_frame import query_frame
from mcp_server.server import mcp

@mcp.tool(name="query_frame_tool")
async def query_frame_tool(
    query: Annotated[str, "Natural language question about video content to analyze"],
    frame_ids: Annotated[Optional[list], "List of specific frame filenames to analyze (e.g., ['video_123.jpg', 'video_456.jpg'])"] = None,
    video_id: Annotated[Optional[str], "Unique video identifier hash for frame retrieval"] = None,
    timestamps: Annotated[Optional[list], "List of time range pairs in HH:MM:SS format, e.g., [['00:07:45', '00:09:44'], ['00:21:22', '00:23:17']]"] = None
) -> str:
    return await query_frame(
        query=query,
        frame_ids=frame_ids,
        video_id=video_id,
        timestamps=timestamps
    )