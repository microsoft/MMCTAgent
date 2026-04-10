"""
MMCT MCP Server Initialization.

This module initializes the FastMCP server instance and defines core routes,
including health monitoring for container orchestration.
"""

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
        JSONResponse: A JSON object indicating the service status, name, and version.
    """
    return JSONResponse({
        "status": "healthy",
        "service": "MMCT Agent MCP Server",
        "version": "1.0.0"
    })