"""
MCP Client for testing the MMCT Agent MCP Server.

This client connects to the MCP server and validates available tools.
Run with: python client.py
"""

import asyncio
from typing import List, Dict, Any
from loguru import logger
from fastmcp import Client


async def list_and_log_tools(client: Client) -> List[str]:
    """
    Fetch and display all available tools from the MCP server.

    Args:
        client (Client): Active MCP client instance.

    Returns:
        List[str]: List of available tool names.
    """
    logger.info("Fetching available tools from the MCP server...")
    tools = await client.list_tools()

    available_tool_names = []
    logger.info("Available tools:")
    for tool in tools:
        available_tool_names.append(tool.name)
        logger.info(f"\nTool Name: {tool.name}")
        logger.info(f"Description: {tool.description}")
        logger.info(f"Input Schema: {tool.inputSchema}")

    return available_tool_names


async def validate_tool(
    client: Client, tool_name: str, arguments: Dict[str, Any]
) -> None:
    """
    Call and validate a specific tool by name.

    Args:
        client (Client): Active MCP client instance.
        tool_name (str): Name of the tool to validate.
        arguments (Dict[str, Any]): Arguments for the tool call.
    """
    logger.info(f"Validating tool: {tool_name}")
    try:
        result = await client.call_tool(name=tool_name, arguments=arguments)
        logger.success(f"Tool '{tool_name}' executed successfully.")
        logger.debug(f"Result: {result}")
    except Exception as e:
        logger.error(f"Failed to execute tool '{tool_name}': {e}")


async def main(tools_to_validate: List[str] = None) -> None:
    """
    Main entry point for connecting with MCP server and validating tools.

    Args:
        tools_to_validate (List[str], optional): List of tool names to validate.
                                                Defaults to available tools.
    """
    if tools_to_validate is None:
        tools_to_validate = ["get_visual_timestamps"]

    try:
        logger.info("Initializing MCP client...")
        client = Client("http://127.0.0.1:8000/mcp")

        async with client:
            # Step 1: Verify connection
            await client.ping()
            logger.success("Connected successfully to the MCP server.")

            # Step 2: Get available tools
            available_tools = await list_and_log_tools(client)

            # Step 3: Validate selected tools
            if "get_visual_timestamps" in tools_to_validate and "get_visual_timestamps" in available_tools:
                await validate_tool(
                    client,
                    "get_visual_timestamps",
                    {
                        "query": "person walking",
                        "top_k": 3,
                        # "video_id": "optional-video-id",  # Optional: filter by video_id
                        # "youtube_url": "https://www.youtube.com/watch?v=...",  # Optional: filter by youtube_url (takes precedence)
                    },
                )

    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with desired tools
    # Supported tools: get_visual_timestamps
    # Input the tools that you want to validate
    asyncio.run(main(tools_to_validate=["get_visual_timestamps"]))
