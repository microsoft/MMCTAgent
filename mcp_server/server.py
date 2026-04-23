"""
MMCT MCP Server Initialization.

This module initializes the FastMCP server instance and defines core routes,
including health monitoring for container orchestration.
"""

import os

from fastmcp import FastMCP
from loguru import logger
from starlette.responses import JSONResponse, HTMLResponse

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


# ---------------------------------------------------------------------------
# Core HTTP routes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# /api — HTTP API reference page
# ---------------------------------------------------------------------------

# Registry of HTTP endpoints shown on /api.  Core routes are always present;
# data routes register themselves at import time via register_api_endpoint().
_api_endpoints: list[dict] = [
    {
        "method": "GET",
        "path": "/",
        "summary": "Health check",
        "description": "Returns service status, version, and build metadata.",
        "tag": "Operations",
    },
    {
        "method": "GET",
        "path": "/readyz",
        "summary": "Readiness probe",
        "description": "Verifies connectivity to video and image pipeline dependencies.",
        "tag": "Operations",
    },
]


def register_api_endpoint(method: str, path: str, summary: str, description: str, tag: str = "Data"):
    """Register an HTTP endpoint to appear on the /api documentation page."""
    _api_endpoints.append({
        "method": method,
        "path": path,
        "summary": summary,
        "description": description,
        "tag": tag,
    })


@mcp.custom_route("/api", methods=["GET"])
async def api_reference(request):
    """Serve an HTML page listing all custom HTTP endpoints."""
    accept = request.headers.get("accept", "")
    if "text/html" not in accept:
        return JSONResponse({"endpoints": _api_endpoints})

    grouped: dict[str, list[dict]] = {}
    for ep in _api_endpoints:
        grouped.setdefault(ep["tag"], []).append(ep)

    sections_html = ""
    for tag, endpoints in grouped.items():
        rows = ""
        for ep in endpoints:
            rows += f"""
            <tr>
              <td><span class="method">{ep["method"]}</span></td>
              <td><code>{ep["path"]}</code></td>
              <td><strong>{ep["summary"]}</strong><br><span class="desc">{ep["description"]}</span></td>
            </tr>"""
        sections_html += f"""
        <h2>{tag}</h2>
        <table>
          <thead><tr><th>Method</th><th>Path</th><th>Description</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MMCT Agent — HTTP API Reference</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; }}
    h1 {{ border-bottom: 2px solid #16213e; padding-bottom: 8px; }}
    h2 {{ color: #16213e; margin-top: 32px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th {{ text-align: left; background: #f0f4f8; padding: 10px 12px; border-bottom: 2px solid #d0d7de; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #e8ecf0; vertical-align: top; }}
    .method {{ background: #1a7f37; color: #fff; padding: 2px 8px; border-radius: 4px;
               font-size: 0.85em; font-weight: 600; }}
    code {{ background: #f0f4f8; padding: 2px 6px; border-radius: 3px; font-size: 0.95em; }}
    .desc {{ color: #57606a; font-size: 0.9em; }}
    .nav {{ margin-bottom: 24px; font-size: 0.95em; }}
    .nav a {{ color: #0969da; text-decoration: none; margin-right: 16px; }}
    .nav a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>MMCT Agent — HTTP API Reference</h1>
  <div class="nav">
    <a href="/docs">MCP Tools Docs</a>
  </div>
  {sections_html}
</body>
</html>"""
    return HTMLResponse(html)