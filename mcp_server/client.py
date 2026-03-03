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
        logger.info(f"Input Schema: {tool.inputSchema}")

    return available_tool_names


async def validate_tool(client: Client, tool_name: str, arguments: Dict[str, Any]) -> None:
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
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Failed to execute tool '{tool_name}': {e}")


async def main(tools_to_validate: List[str] = None) -> None:
    """
    Main entry point for connecting with MCP server and validating tools.

    Args:
        tools_to_validate (List[str], optional): List of tool names to validate.
                                                Defaults to all tools except video_ingestion_tool.
    """
    if tools_to_validate is None:
        tools_to_validate = [
            "video_agent_tool",
            "image_agent_tool",
            "get_context_tool",
            "get_relevant_frames_tool",
            "query_frame_tool",
            "get_object_collection_tool",
            "get_video_summary_tool",
        ]

    try:
        logger.info("Initializing MCP client...")
        client = Client("http://127.0.0.1:8000/mcp")  # change the url accordingly

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
                        "query": "What is this video about?",
                        "use_critic_agent": True,
                        "url": "www.youtube.com/watch?v=2W3BKOSg958",
                    },
                )

            if "image_agent_tool" in tools_to_validate and "image_agent_tool" in available_tools:
                await validate_tool(
                    client,
                    "image_agent_tool",
                    {
                        "query": "Answer the question shown in the image",
                        "image_url": "https://www.doingmaths.co.uk/uploads/8/3/8/9/8389495/4890329_orig.png",
                        "tools": ["vit", "recog"],
                        "use_critic_agent": True,
                        "stream": True,
                    },
                )

            if "get_context_tool" in tools_to_validate and "get_context_tool" in available_tools:
                await validate_tool(
                    client,
                    "get_context_tool",
                    {
                        "query": "What is this video about?",
                        "fields_to_retrieve": [
                            "chapter_transcript",
                            "detailed_summary",
                            "start_time",
                            "end_time",
                            "hash_video_id",
                        ],
                        "top": 3,
                        "url": "www.youtube.com/watch?v=2W3BKOSg958",
                    },
                )

            if (
                "get_relevant_frames_tool" in tools_to_validate
                and "get_relevant_frames_tool" in available_tools
            ):
                await validate_tool(
                    client,
                    "get_relevant_frames_tool",
                    {
                        "query": "Computing gcd of two numbers",
                        "video_id": "2W3BKOSg958",
                        "top_k": 5,
                    },
                )

            if "query_frame_tool" in tools_to_validate and "query_frame_tool" in available_tools:
                await validate_tool(
                    client,
                    "query_frame_tool",
                    {
                        "query": "tell about gcd of two numbers",
                        "video_id": "2W3BKOSg958",
                        "start_time": 60.0,
                        "end_time": 120.0,
                    },
                )

            if (
                "get_object_collection_tool" in tools_to_validate
                and "get_object_collection_tool" in available_tools
            ):
                await validate_tool(
                    client,
                    "get_object_collection_tool",
                    {
                        "object_names": ["Presentation Slide", "board", "text"],
                        "video_id": "2W3BKOSg958",
                    },
                )

            if (
                "get_video_summary_tool" in tools_to_validate
                and "get_video_summary_tool" in available_tools
            ):
                await validate_tool(
                    client,
                    "get_video_summary_tool",
                    {
                        "query": "computing gcd of two numbers",
                        "video_id": "2W3BKOSg958",
                        "top": 1,
                    },
                )

    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with desired tools
    # Supported tools: video_agent_tool, image_agent_tool, get_context_tool,
    #                   get_relevant_frames_tool, query_frame_tool,
    #                   get_object_collection_tool, get_video_summary_tool,
    #                   video_ingestion_tool
    # Input the tools that you want to validate
    asyncio.run(
        main(
            tools_to_validate=[
                "video_agent_tool",
                "image_agent_tool",
                "get_context_tool",
                "get_relevant_frames_tool",
                "query_frame_tool",
                "get_object_collection_tool",
                "get_video_summary_tool",
            ]
        )
    )
