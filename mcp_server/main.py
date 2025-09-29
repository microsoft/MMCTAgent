from mcp_server.server import mcp
from mcp_server.tools.video_agent_tool import video_agent_tool
from mcp_server.tools.image_agent_tool import image_agent_tool
from mcp_server.tools.video_ingestion_tool import video_ingestion_tool
from mcp_server.tools.kb_tool import kb_tool
from mcp_server.tools.get_context_tool import get_context_tool
from mcp_server.tools.get_relevant_frames_tool import get_relevant_frames_tool
from mcp_server.tools.query_frame_tool import query_frame_tool

if __name__=='__main__':
    mcp.run(transport="streamable-http",port=8000,host='0.0.0.0')