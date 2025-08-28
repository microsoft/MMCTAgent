from fastmcp import FastMCP
from loguru import logger

try:
    logger.info("Instiating the FastMCP object")
    mcp = FastMCP(name="MMCT Agent MCP Server")
    logger.info("Successfully created an instance of FastMCP server")
except Exception as e:
    logger.exception(f"Exception occured while creating an instance of FastMCP Server: {e}")
    raise