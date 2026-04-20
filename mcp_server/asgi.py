"""
ASGI application factory for production multi-worker deployment.

This module exposes the MCP server as an ASGI app that Uvicorn can import
directly with multiple workers:

    uvicorn mcp_server.asgi:app --host 0.0.0.0 --port 8000 --workers 4

For local development, use `python -m mcp_server.main` instead (single worker).
"""

import asyncio
import threading

from mcp_server.server import mcp
from mcp_server.tools.image_query_tool import image_query  # noqa: F401 — registers tool
from mcp_server.tools.video_query_tool import video_query  # noqa: F401 — registers tool

from fastmcp_docs import FastMCPDocs
from loguru import logger

# Set up tool documentation.
# Use a separate thread with its own event loop because uvicorn workers
# import this module inside a running event loop, which blocks asyncio.run().
docs = FastMCPDocs(
    mcp,
    title="MMCT Agent MCP Tools",
    description="Interactive documentation for MMCT Agent MCP server tools. "
                "Browse available tools, view input/output schemas, and test calls.",
)


def _setup_docs():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(docs.setup())
    loop.close()


_thread = threading.Thread(target=_setup_docs)
_thread.start()
_thread.join()
logger.info("Tool documentation registered at /docs")

# Create the ASGI application for Uvicorn
app = mcp.http_app(transport="streamable-http")