"""
Image Query Tool for MMCT MCP Server.

This module provides a tool for analyzing images using the MMCT Image Pipeline,
supporting both local file paths and remote URLs.
"""

import os
import aiohttp
import tempfile
from typing import Optional, List, Literal, Dict, Any
from loguru import logger
from mcp_server.server import mcp
from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mmct.image_pipeline.config import ImageAgentProviderConfig
from config.provider_config import get_image_agent_provider

# Singleton provider config — reused across requests.
_provider_config: Optional[ImageAgentProviderConfig] = None


def get_image_provider() -> ImageAgentProviderConfig:
    """Return a singleton ImageAgentProviderConfig."""
    global _provider_config
    if _provider_config is None:
        _provider_config = get_image_agent_provider()
    return _provider_config


async def check_image_health() -> Dict[str, Any]:
    """Check health of the image pipeline's LLM provider."""
    provider = get_image_provider()
    if provider.llm_provider is not None and hasattr(provider.llm_provider, "check_health"):
        return {"llm": await provider.llm_provider.check_health()}
    return {"llm": {"status": "not_configured"}}

@mcp.tool(
    name="image_query",
    description="""
    Analyze an image and answer questions about its content using the MMCT Image Pipeline.
    Supports local file paths or remote URLs.
    
    Args:
        query: The question or instruction about the image.
        image_path: Local path to the image or a public URL.
        use_critic_agent: Whether to use a critic agent for reflective feedback (default: False).
        tools: Optional list of tools to use for analysis (vit, recog, object_detection, ocr).
               If omitted, all tools are used by default.
    """
)
async def image_query(
    query: str,
    image_path: str,
    use_critic_agent: bool = False,
    tools: Optional[List[Literal["vit", "recog", "object_detection", "ocr"]]] = None
) -> dict:
    """
    Analyzes an image and returns a structured response using specialized vision tools.

    Args:
        query (str): The natural language query or instruction.
        image_path (str): The local path or remote URL of the image to analyze.
        use_critic_agent (bool, optional): If True, uses a critic agent to refine the answer. 
            Defaults to False.
        tools (List[str], optional): A list of specific tools to enable. 
            Available: 'vit', 'recog', 'object_detection', 'ocr'. Defaults to all.

    Returns:
        dict: A dictionary containing the analysis result, tokens used, and status.
    """
    save_path = image_path
    is_temp = False
    
    try:
        # Check if image_path is a URL and download if necessary
        if image_path.startswith(("http://", "https://")):
            logger.info(f"Downloading image from URL: {image_path}")
            temp_dir = tempfile.gettempdir()
            save_path = os.path.join(temp_dir, f"mcp_image_{os.path.basename(image_path)}")
            is_temp = True
            
            async with aiohttp.ClientSession() as session:
                async with session.get(image_path) as response:
                    if response.status == 200:
                        with open(save_path, "wb") as f:
                            f.write(await response.read())
                        logger.info(f"Image saved to {save_path}")
                    else:
                        raise Exception(f"Failed to download image source: HTTP {response.status}")

        # Convert string tool names to ImageQnaTools enum members
        tools_enum = [ImageQnaTools(t) for t in tools] if tools else None

        logger.info(f"Executing MMCT ImageAgent with tools: {tools_enum if tools_enum else 'all'}")

        # Initialize and run ImageAgent
        image_agent = ImageAgent(
            image_path=save_path,
            query=query,
            provider=get_image_provider(),
            use_critic_agent=use_critic_agent,
            tools=tools_enum,
            stream=True,
            use_console=True,
            disable_console_log=False
        )
        
        # Execute agent and get structured response
        response = await image_agent()
        
        # Convert Pydantic response model to dictionary for MCP transport
        return response.model_dump()
        
    except Exception as e:
        logger.exception(f"Exception generated during image_query execution: {e}")
        return {"error": str(e), "status": "failed"}
    finally:
        # Clean up temporary files if created
        if is_temp and os.path.exists(save_path):
            os.remove(save_path)
            logger.info(f"Cleaned up temporary resources: {save_path}")
