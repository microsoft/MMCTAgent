from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mcp_server.server import mcp
from typing import Annotated
import os
import uuid
import aiofiles
import aiohttp
from loguru import logger


@mcp.tool(
    name="image_agent_tool",
    description="""The Image Agent Tool (image_agent_tool) enables agents to perform question answering and analysis over images using the MMCT Image Agent Framework.

It accepts an image URL and a natural language query, applies one or more specialized ImageQnaTools (such as OCR, recognition, object detection, or visual transformers), and optionally leverages a critic agent to validate and refine the responses.

This tool internally handles:
- Downloading and preprocessing the image.
- Applying the selected image analysis tools.
- Running multimodal reasoning to answer the query.
- Cleaning up temporary files after execution.

## Input Schema

- image_url (string, required) → Publicly accessible image URL to analyze.
- query (string, required) → Natural language question about the image (e.g., “What text is written on the board?”).
- use_critic_agent (boolean, required) → Whether to enable critic agent for improved reasoning depth.
- tools (list of strings, required) → List of image analysis tools to use. Supported values:
    - VIT → Visual Transformer for image embeddings.
    - OCR → Optical Character Recognition (extracts text).
    - RECOG → Image recognition (entities, scenes).
    - OBJECT_DETECTION → Detects objects in the image.
- stream (boolean, optional, default=False) → Stream intermediate reasoning steps if enabled.
- disable_console_log (boolean, optional, default=False) → Suppress console logs during execution.

## Output

A structured answer to the query about the image, which may include:
- Detected objects or entities.
- Extracted text (OCR results).
- Scene or content recognition.
- Refined reasoning (if critic agent is enabled).
"""
)
async def image_agent_tool(
    image_url: Annotated[str, "Image URL"],
    query: Annotated[str, "query related to image"],
    use_critic_agent: Annotated[bool, "Include critic agent"],
    tools: Annotated[list[str], "ImageQnaTools such as VIT, OCR, RECOG, OBJECT_DETECTION"],
    stream: Annotated[bool, "Enable streaming response (True/False)"] = False,
    disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False,
):
    try:
        save_path = f"image_{str(uuid.uuid4()).split('-')[-1]}.png"
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status == 200:
                    async with aiofiles.open(save_path, "wb") as f:
                        await f.write(await response.read())
                    logger.info(f"Image saved to {save_path}")
                else:
                    logger.warning(f"Failed to download image, status code: {response.status}")
                    raise Exception(f"Failed to download image, status code: {response.status}")

        image_agent = ImageAgent(
            image_path=save_path,
            query=query,
            use_critic_agent=use_critic_agent,
            stream=stream,
            tools=[ImageQnaTools[name.upper()] for name in tools],
            disable_console_log=disable_console_log,
        )
        return await image_agent()
    except Exception as e:
        return f"Exception generated while executing image agent tool: {e}"
    finally:
        if os.path.exists(save_path):
            logger.info(f"Removed the file: {save_path}")
            os.remove(save_path)
