import asyncio
from typing import List, Dict, Any
from loguru import logger
from fastmcp import Client
import datetime

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
                                                Defaults to commonly used tools.
    """
    if tools_to_validate is None:
        tools_to_validate = [
            "video_agent_tool",
            "video_ingestion_tool",
            "image_agent_tool",
        ]

    try:
        logger.info("Initializing MCP client...")
        client = Client("http://127.0.0.1:8000/mcp")    # change the url accordingly

        async with client:
            # Step 1: Verify connection
            await client.ping()
            logger.success("Connected successfully to the MCP server.")

            # Step 2: Get available tools
            available_tools = await list_and_log_tools(client)

            # Step 3: Validate selected tools
            if "video_agent_tool" in tools_to_validate and "video_agent_tool" in available_tools:
                await validate_tool(
                    client,
                    "video_agent_tool",
                    {
                        "query": "user-query",
                        "index_name": "index-name",
                        "top_n": 2,
                        "use_computer_vision_tool": False,
                        "use_critic_agent": True,
                        "stream": True,
                    },
                )

            if "video_ingestion_tool" in tools_to_validate and "video_ingestion_tool" in available_tools:
                await validate_tool(
                    client,
                    "video_ingestion_tool",
                    {
                        "video_url": "video-url",
                        "file_name": "filename.mp4",
                        "index_name": "valid-index-name",
                        "transcription_service": "azure-stt",  # valid transcription services
                        "language": "en-IN",                   # valid language
                        "use_computer_vision_tool": False,
                    },
                )

            if "image_agent_tool" in tools_to_validate and "image_agent_tool" in available_tools:
                await validate_tool(
                    client,
                    "image_agent_tool",
                    {
                        "query": "user-query",
                        "image_url": "image-url",
                        "tools": ["OBJECT_DETECTION", "VIT"],   # valid image agent tools
                        "use_critic_agent": True,
                        "stream": True,
                    },
                )

            if "kb_tool" in tools_to_validate and "kb_tool" in available_tools:
                await validate_tool(
                    client,
                    "kb_tool",
                    {
                        "request": {   # understand the input schema of the kb_tool to understand more about the relevant query and filter parameters
                            "query": "relevant-user-query",
                            "query_type": "available query-type",
                            "index_name": "index-name",
                            "k": 10,
                            "filters": {
                                "category": "relevant-category",    # relevant category         
                                "hash_video_id": "hash-video-id",   # hash video id filter
                                "time_from": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                            }
                        }
                    }
                )

    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with desired tools
    # supported tools: kb_tool, video_agent_tool, video_ingestion_tool, image_agent_tool
    # input the tools that you have to validate
    asyncio.run(main(tools_to_validate=["kb_tool"]))