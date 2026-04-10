"""
Main entry point for the MMCT MCP Server.

This script imports the initialized MCP server and all registered tools,
then starts the server using the Streamable HTTP transport.
"""

from mcp_server.server import mcp
from mcp_server.tools.image_query_tool import image_query
from mcp_server.tools.video_query_tool import video_query

if __name__ == '__main__':
    # Initialize and start the server
    logger_name = "MMCT Agent MCP Server"
    from loguru import logger
    logger.info(f"Starting {logger_name}...")
    
    # Run in streamable HTTP mode
    # Access via http://0.0.0.0:8000/mcp
    mcp.run(transport="streamable-http", port=8000, host='0.0.0.0')
