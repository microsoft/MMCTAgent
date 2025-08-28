from mcp_server.server import mcp
from mcp_server.tools.video_agent_tool import video_agent_tool
from mcp_server.tools.image_agent_tool import image_agent_tool
from mcp_server.tools.video_ingestion_tool import video_ingestion_tool

if __name__=='__main__':
    mcp.run(transport="streamable-http",port=8000,host='0.0.0.0')