"""
MMCT MCP Server Initialization.

This module initializes the FastMCP server instance and defines core routes,
including health monitoring for container orchestration.
"""

import os

from fastmcp import FastMCP
from loguru import logger
from starlette.responses import JSONResponse

from mmct.utils.logging_config import log_manager

# Initialise logging sinks — console always, Azure Monitor when configured.
log_manager.enable_console()
log_manager.enable_azure_monitor()

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


@mcp.custom_route("/readyz", methods=["GET"])
async def readiness_check(request):
    """
    Readiness probe that verifies connectivity to external dependencies
    by delegating to the singleton pipeline and image provider health checks.

    Each provider implements its own ``check_health()`` method, so this
    endpoint works regardless of the underlying database or LLM backend.
    """
    from mcp_server.tools.video_query_tool import get_pipeline
    from mcp_server.tools.image_query_tool import check_image_health

    checks = {}
    all_ok = True

    # --- Video pipeline (graph_state) providers ---
    try:
        pipeline = get_pipeline("graph_state")
        video_health = await pipeline.check_health()
        checks["video_pipeline"] = video_health
        for v in video_health.values():
            if isinstance(v, dict) and v.get("status") not in ("ok", "not_configured"):
                all_ok = False
    except Exception as e:
        checks["video_pipeline"] = {"status": "error", "error": str(e)}
        all_ok = False

    # --- Image pipeline LLM provider ---
    try:
        image_health = await check_image_health()
        checks["image_pipeline"] = image_health
        for v in image_health.values():
            if isinstance(v, dict) and v.get("status") not in ("ok", "not_configured"):
                all_ok = False
    except Exception as e:
        checks["image_pipeline"] = {"status": "error", "error": str(e)}
        all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        {"ready": all_ok, "checks": checks},
        status_code=status_code,
    )