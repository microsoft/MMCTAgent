"""
Main entry point for the MCP Server.

Run with: python -m mcp_server.main
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from lively/.env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
logger.info(f"Loaded environment from: {env_path}")

# Import server and tools
from mcp_server.server import mcp
from mcp_server.tools.get_visual_timestamps import get_visual_timestamps

if __name__ == '__main__':
    logger.info("Starting MCP Server...")
    logger.info(f"Search Endpoint: {os.getenv('SEARCH_SERVICE_ENDPOINT')}")
    logger.info(f"Index Name: {os.getenv('SEARCH_INDEX_NAME', 'video-keyframes-index')}")
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="streamable-http", port=port, host='0.0.0.0')