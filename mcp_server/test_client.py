"""
Test Client for MMCT MCP Server.

This script provides a simple client to verify that the MMCT MCP server
is running correctly and that its tools (image_query, video_query) 
are registered and functional.
"""

import asyncio
from typing import List, Dict, Any
from loguru import logger
from fastmcp import Client

async def list_tools(client: Client):
    """
    Retrieves and logs the list of available tools from the MCP server.

    Args:
        client (Client): The FastMCP client instance.
    """
    logger.info("Fetching available tools...")
    tools = await client.list_tools()
    for tool in tools:
        logger.info(f"Tool: {tool.name}")
        logger.info(f"  Description: {tool.description}")
        logger.info(f"  Schema: {tool.inputSchema}")

async def test_tool(client: Client, tool_name: str, arguments: Dict[str, Any]):
    """
    Executes a specific tool on the MCP server with the provided arguments.

    Args:
        client (Client): The FastMCP client instance.
        tool_name (str): The name of the tool to execute.
        arguments (Dict[str, Any]): The arguments to pass to the tool.
    """
    logger.info(f"Testing tool: {tool_name} with args: {arguments}")
    try:
        result = await client.call_tool(name=tool_name, arguments=arguments)
        logger.success(f"Tool '{tool_name}' executed successfully.")
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Failed to execute tool '{tool_name}': {e}")

async def main():
    """
    Main asynchronous entry point for the test client.
    Connects to the server, lists tools, and runs functional test cases.
    """
    # Note: Ensure the server is running on port 8000 before executing this script
    # To run server: python -m mcp_server.main
    server_url = "http://127.0.0.1:8000/mcp"
    
    try:
        logger.info(f"Connecting to MCP server at {server_url}...")
        client = Client(server_url)
        
        async with client:
            await client.ping()
            logger.success("Connected to MCP server.")
            
            # List all registered tools to verify server registration
            await list_tools(client)
            
            # 1. Test image_query
            # Replace with a valid local path or remote URL for actual environmental testing
            await test_tool(
                client, 
                "image_query", 
                {
                    "query": "What is in this image?",
                    "image_path": "https://www.w3.org/Graphics/PNG/nurbcup2si.png",
                    "use_critic_agent": False,
                    "tools": ["vit", "ocr"]
                }
            )
            
            # 2. Test video_query (graph_state - default)
            # Uses the deterministic state machine for faster extraction.
            await test_tool(
                client,
                "video_query",
                {
                    "query": "What are the main topics discussed?",
                    "video_id": "58550",
                    "mode": "graph_state"
                }
            )

            # 3. Test video_query (graph_agent)
            # Uses the agentic swarm for complex reasoning and synthesis.
            await test_tool(
                client,
                "video_query",
                {
                    "query": "Who is the presenter?",
                    "video_id": "58550",
                    "mode": "graph_agent"
                }
            )
            
    except Exception as e:
        logger.error(f"Test client initialization or execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
