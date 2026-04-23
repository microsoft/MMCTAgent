"""
Main entry point for the MMCT MCP Server.

This script imports the initialized MCP server and all registered tools,
then starts the server using the Streamable HTTP transport.

Endpoints:
    /       — Health check (JSON)
    /api    — HTTP API reference
    /mcp    — MCP protocol (streamable-HTTP)
    /docs   — Interactive MCP tool documentation (Swagger-like UI)
"""

import asyncio
import os

from mcp_server.server import mcp
from mcp_server.tools.image_query_tool import image_query
from mcp_server.tools.video_query_tool import video_query

if os.environ.get("ENABLE_DATA_APIS", "").lower() == "true":
    from mcp_server.tools.data_routes import get_transcript, get_frames, get_video_ids  # noqa: F401

if __name__ == '__main__':
    from loguru import logger
    from fastmcp_docs import FastMCPDocs

    logger.info("Starting MMCT Agent MCP Server...")

    # Set up interactive tool documentation at /docs
    docs = FastMCPDocs(
        mcp,
        title="MMCT Agent MCP Tools",
        description="Interactive documentation for MMCT Agent MCP server tools. "
                    "Browse available tools, view input/output schemas, and test calls.",
        base_url=os.environ.get("BASE_URL", ""),
    )
    asyncio.run(docs.setup())
    logger.info("Tool documentation available at /docs")

    # Run in streamable HTTP mode
    # Access via http://0.0.0.0:8000/mcp
    mcp.run(transport="streamable-http", port=8000, host='0.0.0.0')
