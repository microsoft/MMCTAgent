"""
Helper functions specific to video ingestion pipeline.
"""

import os
import subprocess
import aiofiles
from loguru import logger
from mmct.video_pipeline.utils.ai_search_client import AISearchClient


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


async def split_video_if_needed(video_path: str) -> tuple[list[str], list[str]]:
    """
    Split video into two parts if duration >= 30 minutes.

    Args:
        video_path: Path to the input video file

    Returns:
        tuple: (list of video paths, list of corresponding hash suffixes)
               - If split: ([part_A_path, part_B_path], ['', 'B'])
               - If not split: ([original_path], [''])

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails
        RuntimeError: If video splitting fails or output files not found
    """
    try:
        duration = await get_video_duration(video_path)

        # Check if video is >= 30 minutes (1800 seconds)
        if duration < 1800:
            logger.info("Video duration is less than 30 minutes, no splitting needed")
            return [video_path], ['']

        logger.info("Video duration is >= 30 minutes, splitting video into two parts")

        # Get video file info
        video_name, video_ext = os.path.splitext(os.path.basename(video_path))

        # Get media folder for output paths
        from mmct.video_pipeline.utils.helper import get_media_folder
        media_folder = await get_media_folder()

        # Calculate split point (half duration)
        split_point = duration / 2

        # Define output paths in media folder
        part_a_path = os.path.join(media_folder, f"{video_name}_part_A{video_ext}")
        part_b_path = os.path.join(media_folder, f"{video_name}_part_B{video_ext}")

        # Split video using ffmpeg
        # Part A: from start to middle
        cmd_a = [
            'ffmpeg', '-i', video_path, '-t', str(split_point),
            '-c', 'copy', '-avoid_negative_ts', 'make_zero', part_a_path, '-y'
        ]

        # Part B: from middle to end
        cmd_b = [
            'ffmpeg', '-i', video_path, '-ss', str(split_point),
            '-c', 'copy', '-avoid_negative_ts', 'make_zero', part_b_path, '-y'
        ]

        logger.info("Splitting video into Part A...")
        subprocess.run(cmd_a, capture_output=True, text=True, check=True)
        logger.info(f"Part A created successfully: {part_a_path}")

        logger.info("Splitting video into Part B...")
        subprocess.run(cmd_b, capture_output=True, text=True, check=True)
        logger.info(f"Part B created successfully: {part_b_path}")

        # Verify both parts were created
        if not os.path.exists(part_a_path) or not os.path.exists(part_b_path):
            raise RuntimeError("Video splitting failed - output files not found")

        logger.info(f"Video successfully split into:\n  Part A: {part_a_path}\n  Part B: {part_b_path}")
        return [part_a_path, part_b_path], ['', 'B']

    except subprocess.CalledProcessError as e:
        logger.error(f"Error splitting video: {e}")
        logger.error(f"ffmpeg stderr: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during video splitting: {e}")
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


def split_transcript_by_segments(srt_content: str, segment_count: int = 2) -> list:
    """
    Split transcript content into segments for parallel processing.
    Resets timestamps for each chunk to start from 0 (matching video chunks created with FFmpeg).

    Args:
        srt_content: Original SRT content
        segment_count: Number of segments to split into (default: 2)

    Returns:
        list: List of SRT content strings for each segment
    """
    segments = parse_srt_timestamps(srt_content)

    if not segments:
        return [srt_content]

    # Split segments into chunks
    chunk_size = len(segments) // segment_count
    transcript_chunks = []

    for i in range(segment_count):
        start_idx = i * chunk_size
        if i == segment_count - 1:
            # Last chunk gets remaining segments
            end_idx = len(segments)
        else:
            end_idx = (i + 1) * chunk_size

        chunk_segments = segments[start_idx:end_idx]

        if not chunk_segments:
            continue

        # Calculate time offset for chunks after the first one (Part B, C, etc.)
        # Part A (i == 0) keeps original timestamps, others reset to start from 0
        if i == 0:
            # Part A: Keep original timestamps
            time_offset = 0
        else:
            # Part B and beyond: Reset timestamps to start from 0
            # This matches what FFmpeg does with -avoid_negative_ts make_zero
            time_offset = chunk_segments[0]['start_time']

        # Rebuild SRT format for this chunk
        chunk_srt = ""
        for j, segment in enumerate(chunk_segments, 1):
            # Convert seconds back to SRT timestamp format
            def seconds_to_timestamp(seconds):
                # Ensure non-negative seconds
                seconds = max(0, seconds)
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = seconds % 60
                return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

            # Apply offset (0 for Part A, calculated offset for Part B+)
            adjusted_start_time = segment['start_time'] - time_offset
            adjusted_end_time = segment['end_time'] - time_offset

            start_timestamp = seconds_to_timestamp(adjusted_start_time)
            end_timestamp = seconds_to_timestamp(adjusted_end_time)

            chunk_srt += f"{j}\n{start_timestamp} --> {end_timestamp}\n{segment['text']}\n\n"

        transcript_chunks.append(chunk_srt.strip())

    return transcript_chunks


async def check_video_already_ingested(hash_id: str, index_name: str) -> bool:
    """
    Check if a video with the given hash_id already exists in the search index.

    Args:
        hash_id: The hash ID of the video to check
        index_name: The name of the search index to check

    Returns:
        bool: True if video already exists, False otherwise
    """
    try:
        # Get credentials based on environment configuration
        if os.environ.get("MANAGED_IDENTITY", "FALSE").upper() == "TRUE":
            from azure.identity import AzureCliCredential, DefaultAzureCredential
            try:
                cli_credential = AzureCliCredential()
                cli_credential.get_token("https://search.azure.com/.default")
                credential = cli_credential
            except Exception:
                credential = DefaultAzureCredential()
        else:
            from azure.core.credentials import AzureKeyCredential
            key = os.getenv("SEARCH_SERVICE_KEY")
            if key is None:
                raise Exception("SEARCH_SERVICE_KEY is missing for Azure AI Search!")
            credential = AzureKeyCredential(key)

        # Create AI Search client
        index_client = AISearchClient(
            endpoint=os.getenv("SEARCH_ENDPOINT"),
            index_name=index_name,
            credential=credential
        )

        # Check if document exists
        exists = await index_client.check_if_exists(hash_id=hash_id)
        await index_client.close()

        return exists

    except Exception as e:
        logger.warning(f"{e}")
        # In case of error, return False to proceed with ingestion
        return False
