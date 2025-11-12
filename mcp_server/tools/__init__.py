"""
MCP Server Tools

This module provides MCP tool wrappers for the MMCT agent framework.
"""

# Import all MCP tools to register them with the server
from mcp_server.tools.video_ingestion_tool import video_ingestion_tool
from mcp_server.tools.kb_tool import kb_tool
from mcp_server.tools.video_agent_tool import video_agent_tool
from mcp_server.tools.image_agent_tool import image_agent_tool
from mcp_server.tools.get_context_tool import get_context_tool
from mcp_server.tools.query_frame_tool import query_frame_tool
from mcp_server.tools.get_relevant_frames_tool import get_relevant_frames_tool
from mcp_server.tools.query_federator_tool import query_federator_tool

__all__ = [
    "video_ingestion_tool",
    "kb_tool",
    "video_agent_tool",
    "image_agent_tool",
    "get_context_tool",
    "query_frame_tool",
    "get_relevant_frames_tool",
    "query_federator_tool",
]
