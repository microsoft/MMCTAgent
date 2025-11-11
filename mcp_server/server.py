from fastmcp import FastMCP
from loguru import logger
from starlette.responses import JSONResponse

try:
    logger.info("Instiating the FastMCP object")
    mcp = FastMCP(name="MMCT Agent MCP Server")
    logger.info("Successfully created an instance of FastMCP server")
except Exception as e:
    logger.exception(f"Exception occured while creating an instance of FastMCP Server: {e}")
    raise

# Health probe endpoint
@mcp.custom_route("/", methods=["GET"])
async def health_check(request):
    """Health probe endpoint for container orchestration and monitoring"""
    return JSONResponse({
        "status": "healthy",
        "service": "MMCT Agent MCP Server",
        "version": "1.0.0"
    })