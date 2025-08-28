from mcp_server.server import mcp
from typing import Optional, Annotated
from mmct.video_pipeline import VideoAgent

@mcp.tool(name="video_agent_tool")
async def video_agent_tool(
    query: str,
    index_name: str,
    video_id: Optional[str] = None,
    youtube_url: Optional[str] = None,
    top_n: int = 1,
    use_computer_vision_tool: bool = True,
    use_critic_agent: Optional[bool] = True,
    stream: Optional[bool] = False,
    disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False
):
    video_agent = VideoAgent(
        query=query,
        index_name=index_name,
        video_id=video_id,
        youtube_url=youtube_url,
        top_n=top_n,
        use_computer_vision_tool=use_computer_vision_tool,
        use_critic_agent=use_critic_agent,
        stream=stream,
        disable_console_log=disable_console_log
    )   
    return await video_agent()