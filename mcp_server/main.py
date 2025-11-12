from mcp_server.server import mcp

from loguru import logger
# from mcp_server.tools.query_federator_tool import query_federator_tool
from mcp_server.tools.video_agent_tool import video_agent_tool
# from mcp_server.tools.image_agent_tool import image_agent_tool
# from mcp_server.tools.video_ingestion_tool import video_ingestion_tool
# from mcp_server.tools.kb_tool import kb_tool
# from mcp_server.tools.get_context_tool import get_context_tool
# from mcp_server.tools.get_relevant_frames_tool import get_relevant_frames_tool
# from mcp_server.tools.query_frame_tool import query_frame_tool

if __name__=='__main__':
    logger.info("Starting MMCT Agent MCP Server with HTTP SSE streaming on http://0.0.0.0:8000/mcp")
    mcp.run(
        transport="streamable-http",  # Server-Sent Events for HTTP streaming
        port=8000,
        host='0.0.0.0',
        path="/mcp"
    )