"""
This is a get_summary_n_transcript tool which provide the summary with the transcript of video.
"""
# Importing Libraries
import os
from typing_extensions import Annotated
from mmct.video_pipeline.utils.helper import get_media_folder
from loguru import logger

async def get_summary_n_transcript(video_id:Annotated[str,'video id'])->str:
    logger.info("Utilizing the Get Summary and Transcript")
    base_dir = await get_media_folder()
    summary_path = os.path.join(base_dir, f"{video_id}.json")
    if not os.path.exists(summary_path):
        logger.error(f"File path do not exists: {summary_path}")
    with open(summary_path, 'r', encoding="utf-8") as file:
        return file.read()