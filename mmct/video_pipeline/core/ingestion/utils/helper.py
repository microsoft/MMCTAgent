"""
Helper functions specific to video ingestion pipeline.
"""

import subprocess
import aiofiles
from loguru import logger
from typing import Any


async def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        float: Video duration in seconds

    Raises:
        subprocess.CalledProcessError: If ffprobe command fails
        ValueError: If duration cannot be parsed
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"Video duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        return duration
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting video duration: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error parsing video duration: {e}")
        raise



async def load_srt(path: str) -> str:
    """
    Asynchronously load the full contents of an SRT (SubRip Subtitle) transcript file.

    Args:
        path: Path to the .srt transcript file

    Returns:
        str: The complete content of the SRT file with original formatting
    """
    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        content = await f.read()
    return content.strip()


def parse_srt_timestamps(srt_content: str) -> list:
    """
    Parse SRT content to extract timestamps and text segments.

    Args:
        srt_content: SRT file content as string

    Returns:
        list: List of dictionaries with 'start_time', 'end_time', 'text' keys
    """
    segments = []
    blocks = srt_content.strip().split('\n\n')

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Extract timestamp line (second line)
        timestamp_line = lines[1]
        if '-->' not in timestamp_line:
            continue

        # Parse timestamps
        start_time_str, end_time_str = timestamp_line.split(' --> ')

        # Convert timestamp to seconds
        def timestamp_to_seconds(timestamp_str):
            timestamp_str = timestamp_str.replace(',', '.')
            h, m, s = timestamp_str.split(':')
            return int(h) * 3600 + int(m) * 60 + float(s)

        start_time = timestamp_to_seconds(start_time_str.strip())
        end_time = timestamp_to_seconds(end_time_str.strip())

        # Extract text (lines after timestamp)
        text = '\n'.join(lines[2:])

        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': text
        })

    return segments


async def check_video_already_ingested(hash_id: str, search_provider: Any) -> bool:
    """
    Check if a video with the given hash_id already exists in the search index.

    Args:
        hash_id: The hash ID of the video to check
        index_name: The name of the search index to check

    Returns:
        bool: True if video already exists, False otherwise
    """
    try:
        
        # First check if index exists
        index_exists = await search_provider.index_exists()
        if not index_exists:
            logger.info(f"Index '{search_provider.index_name}' does not exist yet, skipping duplicate check")
            return False

        # Check if document exists
        exists = await search_provider.check_is_document_exist(
            hash_id=hash_id,
        )

        return exists

    except Exception as e:
        logger.warning(f"Error checking if video already ingested: {e}")
        # In case of error, return False to proceed with ingestion
        return False
