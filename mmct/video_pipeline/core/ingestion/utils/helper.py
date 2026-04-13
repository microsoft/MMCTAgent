"""
Helper functions specific to video ingestion pipeline.
"""

import os
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



async def load_vtt(path: str) -> str:
    """
    Asynchronously load a WebVTT (.vtt) transcript file and convert it to SRT format.

    WebVTT cues use '.' as the millisecond separator and may omit the hours
    component, whereas SRT uses ',' and always includes hours.  This function
    normalises the content so that downstream SRT-based parsers work unchanged.

    Args:
        path: Path to the .vtt transcript file

    Returns:
        str: The transcript content converted to SRT format
    """
    import re

    async with aiofiles.open(path, mode='r', encoding='utf-8') as f:
        content = await f.read()

    # Strip the WEBVTT header and any metadata lines before the first cue
    content = re.sub(r'^WEBVTT[^\n]*\n', '', content, count=1)
    # Remove NOTE blocks
    content = re.sub(r'NOTE\s[^\n]*(?:\n(?!\n)[^\n]*)*\n?', '', content)
    # Remove style blocks
    content = re.sub(r'STYLE\s[^\n]*(?:\n(?!\n)[^\n]*)*\n?', '', content)

    blocks = content.strip().split('\n\n')
    srt_blocks: list[str] = []
    seq = 0

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        # Find the timestamp line (contains '-->')
        ts_idx = None
        for i, line in enumerate(lines):
            if '-->' in line:
                ts_idx = i
                break
        if ts_idx is None:
            continue

        timestamp_line = lines[ts_idx]
        text_lines = lines[ts_idx + 1:]
        if not text_lines:
            continue

        # Strip any cue settings after the end timestamp (e.g. "align:start")
        parts = timestamp_line.split('-->')
        start_raw = parts[0].strip()
        end_with_settings = parts[1].strip()
        end_raw = end_with_settings.split()[0]

        # Normalise VTT timestamps to SRT format (HH:MM:SS,mmm)
        def _normalise_ts(ts: str) -> str:
            ts = ts.replace('.', ',')
            # Add hours if missing (MM:SS,mmm -> 00:MM:SS,mmm)
            if ts.count(':') == 1:
                ts = '00:' + ts
            return ts

        start_srt = _normalise_ts(start_raw)
        end_srt = _normalise_ts(end_raw)

        # Strip VTT cue tags like <v Speaker>, <c>, etc.
        cleaned_text = '\n'.join(
            re.sub(r'<[^>]+>', '', line) for line in text_lines
        )

        seq += 1
        srt_blocks.append(f"{seq}\n{start_srt} --> {end_srt}\n{cleaned_text}")

    return '\n\n'.join(srt_blocks)


async def load_transcript(path: str) -> str:
    """
    Load a transcript file, auto-detecting format from the file extension.

    Supports .srt (SubRip) and .vtt (WebVTT) files.  VTT files are
    transparently converted to SRT format.

    Args:
        path: Path to the transcript file (.srt or .vtt)

    Returns:
        str: The transcript content in SRT format

    Raises:
        ValueError: If the file extension is not supported
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.vtt':
        return await load_vtt(path)
    if ext == '.srt':
        return await load_srt(path)
    raise ValueError(f"Unsupported transcript format '{ext}'. Expected .srt or .vtt")


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


def parse_vtt_timestamps(vtt_content: str) -> list:
    """
    Parse WebVTT content to extract timestamps and text segments.

    Handles VTT-specific features: optional cue identifiers, '.' as the
    millisecond separator, optional hours component, cue settings, and
    inline tags like ``<v Speaker>``.

    Args:
        vtt_content: VTT file content as string

    Returns:
        list: List of dictionaries with 'start_time', 'end_time', 'text' keys
              (times in seconds as float)
    """
    import re

    segments: list[dict] = []
    # Strip header
    content = re.sub(r'^WEBVTT[^\n]*\n', '', vtt_content, count=1)
    blocks = content.strip().split('\n\n')

    for block in blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')

        # Find the line with '-->'
        ts_idx = None
        for i, line in enumerate(lines):
            if '-->' in line:
                ts_idx = i
                break
        if ts_idx is None:
            continue

        timestamp_line = lines[ts_idx]
        text_lines = lines[ts_idx + 1:]
        if not text_lines:
            continue

        parts = timestamp_line.split('-->')
        start_raw = parts[0].strip()
        end_raw = parts[1].strip().split()[0]

        def _ts_to_seconds(ts: str) -> float:
            ts = ts.replace(',', '.')
            components = ts.split(':')
            if len(components) == 2:
                m, s = components
                return int(m) * 60 + float(s)
            h, m, s = components
            return int(h) * 3600 + int(m) * 60 + float(s)

        start_time = _ts_to_seconds(start_raw)
        end_time = _ts_to_seconds(end_raw)

        text = '\n'.join(re.sub(r'<[^>]+>', '', line) for line in text_lines)

        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'text': text,
        })

    return segments


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
