"""
MMCT MCP Server Initialization.

This module initializes the FastMCP server instance and defines core routes,
including health monitoring for container orchestration.
"""

import os

from fastmcp import FastMCP
from loguru import logger
from starlette.responses import JSONResponse

try:
    logger.info("Instantiating the FastMCP object")
    mcp = FastMCP(name="MMCT Agent MCP Server")
    logger.info("Successfully created an instance of FastMCP server")
except Exception as e:
    logger.exception(f"Exception occurred while creating an instance of FastMCP Server: {e}")
    raise

@mcp.custom_route("/", methods=["GET"])
async def health_check(request):
    """
    Health probe endpoint for container orchestration and monitoring.

    Returns:
        JSONResponse: A JSON object indicating the service status, name, version,
        and build metadata (populated by CI/CD pipeline).
    """
    return JSONResponse({
        "status": "healthy",
        "service": "MMCT Agent MCP Server",
        "version": "1.0.0",
        "build": {
            "sha": os.getenv("BUILD_SHA", "local"),
            "run_id": os.getenv("BUILD_RUN_ID", "n/a"),
            "run_url": os.getenv("BUILD_RUN_URL", "n/a"),
            "timestamp": os.getenv("BUILD_TIMESTAMP", "n/a"),
        }
    })