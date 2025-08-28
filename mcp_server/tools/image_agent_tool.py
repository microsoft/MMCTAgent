from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mcp_server.server import mcp
from typing import Annotated
import os
import uuid
import aiofiles
import aiohttp
from loguru import logger


@mcp.tool(name="image_agent_tool")
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
