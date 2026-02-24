from mcp_server.server import mcp
from mcp_server.config import get_ingestion_provider
from typing import Annotated, Optional
from mmct.video_pipeline.core.ingestion.ingestion_pipeline import (
    IngestionPipeline,
    Languages,
)
from mmct.video_pipeline.utils.helper import get_file_hash
import os
import aiohttp
import aiofiles
from loguru import logger


@mcp.tool(
    name="video_ingestion_tool",
    description="""The Video Ingestion Tool (video_ingestion_tool) enables agents to ingest video content into the MMCT Video Pipeline, preparing it for downstream search and question-answering tasks.

This tool downloads the video from a given URL, processes it through the ingestion pipeline, and enriches it with transcripts, summaries, visual frames, and metadata. It supports multiple languages, transcription services, optional computer vision analysis, and configurable frame stacking for efficient indexing.

Once ingested, the video data is stored in the specified search index, making it available for later retrieval and multimodal reasoning via other MMCT tools.

## Input Schema
- video_url (string, required) → Publicly accessible video URL.
- file_name (string, required) → Local file name for temporary storage.
- language (enum, required) → Video language (from Languages enum).
- url (string, optional) → Source URL (if available).
- transcript_url (string, optional) → URL to download transcript file from.
- transcript_file_name (string, optional) → Local filename for the downloaded transcript.
- disable_console_log (boolean, optional, default=False) → Suppress console logging.
- hash_video_id (string, optional) → Unique hash identifier for the video.
- frame_stacking_grid_size (int, optional, default=4) → Grid size for frame stacking. Values >1 enable stacking, 1 disables.

## Output

No direct response is returned. The ingestion pipeline enriches and indexes the video in the specified knowledge base index, enabling later use by video_agent_tool and other MMCT flows.
""",
)
async def video_ingestion_tool(
    video_url: Annotated[str, "Video URL"],
    file_name: Annotated[str, "File name of the video"],
    language: Languages,
    url: Optional[str] = None,
    transcript_url: Optional[str] = None,
    transcript_file_name: Optional[str] = None,
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

        if transcript_url and transcript_file_name:
            async with aiohttp.ClientSession() as session:
                async with session.get(transcript_url) as response:
                    if response.status == 200:
                        async with aiofiles.open(transcript_file_name, "wb") as f:
                            await f.write(await response.read())
                        logger.info(f"Transcript saved to {transcript_file_name}")
                    else:
                        logger.warning(
                            f"Failed to download transcript, status code: {response.status}"
                        )
                        raise Exception(
                            f"Failed to download transcript, status code: {response.status}"
                        )

        video_path = os.path.join(os.getcwd(), file_name)
        if not hash_video_id:
            hash_video_id = await get_file_hash(video_path)

        ingestion_tool = IngestionPipeline(
            video_path=video_path,
            video_id=hash_video_id,
            provider=get_ingestion_provider(),
            language=language,
            url=url,
            transcript_path=(
                os.path.join(os.getcwd(), transcript_file_name) if transcript_file_name else None
            ),
            disable_console_log=disable_console_log,
            frame_stacking_grid_size=frame_stacking_grid_size,
        )

        await ingestion_tool.run()
    except Exception as e:
        raise e
    finally:
        if os.path.exists(file_name):
            logger.info(f"Removed the file: {file_name}")
            os.remove(file_name)
