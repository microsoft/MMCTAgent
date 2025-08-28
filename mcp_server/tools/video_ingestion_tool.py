from mcp_server.server import mcp
from typing import Annotated, Optional
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import (
    IngestionPipeline,
    Languages,
)
import os
import aiohttp
import aiofiles
from loguru import logger


@mcp.tool(name="video_ingestion_tool")
async def video_ingestion_tool(
    video_url: Annotated[str, "Video URL"],
    file_name: Annotated[str, "File name of the video"],
    index_name: str,
    language: Languages,
    transcription_service: Optional[str] = None,
    youtube_url: Optional[str] = None,
    transcript_path: Optional[str] = None,
    use_computer_vision_tool: Optional[bool] = False,
    disable_console_log: Annotated[bool, "boolean flag to disable console logs"] = False,
    hash_video_id: Annotated[str, "unique Hash Video Id"] = None,
    frame_stacking_grid_size: Annotated[
        int, "Grid size for frame stacking (>1 enables stacking, 1 disables)"
    ] = 4,
):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url) as response:
                if response.status == 200:
                    async with aiofiles.open(file_name, "wb") as f:
                        await f.write(await response.read())
                    logger.info(f"Video saved to {file_name}")
                else:
                    logger.warning(f"Failed to download video, status code: {response.status}")
                    raise Exception(f"Failed to download video, status code: {response.status}")

        ingestion_tool = IngestionPipeline(
            video_path=os.path.join(os.getcwd(), file_name),
            index_name=index_name,
            language=language,
            transcription_service=transcription_service,
            youtube_url=youtube_url,
            transcript_path=transcript_path,
            use_computer_vision_tool=use_computer_vision_tool,
            disable_console_log=disable_console_log,
            hash_video_id=hash_video_id,
            frame_stacking_grid_size=frame_stacking_grid_size,
        )

        await ingestion_tool()
    except Exception as e:
        raise e
    finally:
        if os.path.exists(file_name):
            logger.info(f"Removed the file: {file_name}")
            os.remove(file_name)
