"""
Utility functions for the ingestion pipeline.
"""

import os
import hashlib
import logging
import aiofiles
from typing import Optional

logger = logging.getLogger(__name__)


async def get_file_hash(file_path: str, hash_algorithm: str = "sha256", suffix: str = "") -> str:
    """
    Generate a hash for a file asynchronously.

    Args:
        file_path: Path to the file
        hash_algorithm: Hash algorithm to use (default: sha256)
        suffix: Optional suffix to append to the hash

    Returns:
        Hash as hexadecimal string with optional suffix
    """
    try:
        hash_func = hashlib.new(hash_algorithm)

        async with aiofiles.open(file_path, "rb") as file:
            while True:
                chunk = await file.read(8192)  # Read chunks asynchronously
                if not chunk:
                    break
                hash_func.update(chunk)

        hash_id = hash_func.hexdigest() + suffix
        logger.info(f"Hash Id Generated: {hash_id}")
        return hash_id
    except Exception as e:
        logger.exception(f"Error generating hash id of video, error: {e}")
        raise


async def get_media_folder(custom_path: Optional[str] = None) -> str:
    """
    Get or create media folder for temporary files.

    Args:
        custom_path: Optional custom path for media folder

    Returns:
        Path to media folder
    """
    if custom_path:
        media_folder = custom_path
    else:
        media_folder = os.path.join(os.getcwd(), "media")

    os.makedirs(media_folder, exist_ok=True)
    return media_folder
