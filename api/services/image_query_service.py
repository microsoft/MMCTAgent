"""Service layer for the image query endpoint."""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import HTTPException
from loguru import logger

from mmct.image_pipeline.agents.image_agent import ImageAgent, ImageQnaTools
from mmct.image_pipeline.config import ImageAgentProviderConfig
from mmct.utils.error_handler import ConfigurationException, ProviderException


async def run_image_query(
    image_path: Path,
    query: str,
    tool_names: List[str],
    use_critic_agent: bool,
    provider_config: ImageAgentProviderConfig,
) -> Dict[str, Any]:
    """
    Execute the ImageAgent for a single image and query.

    Converts string tool names (from ImageToolName schema enum) to the
    corresponding ImageQnaTools members by name lookup, then runs the agent.
    Returns a dict compatible with ImageQueryResponse.
    """
    try:
        try:
            tools = [ImageQnaTools[name] for name in tool_names]
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown tool name: {exc}")

        if not tools:
            raise HTTPException(status_code=400, detail="At least one tool must be selected.")

        agent = ImageAgent(
            image_path=str(image_path),
            query=query,
            provider=provider_config,
            use_critic_agent=use_critic_agent,
            stream=False,
            tools=tools,
            disable_console_log=True,
            use_console=False,
        )

        response = await agent()  # returns ImageAgentResponse

        return {
            "response": response.response,
            "input_tokens": response.tokens.input_token,
            "output_tokens": response.tokens.output_token,
        }

    except HTTPException:
        raise
    except (ConfigurationException, ProviderException) as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")
    except Exception as exc:
        logger.exception("Image query agent failed")
        raise HTTPException(status_code=500, detail=f"Image query failed: {exc}")
