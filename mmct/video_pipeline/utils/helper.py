import hashlib
import os
from typing import Optional


async def get_media_folder(base_dir: Optional[str] = None) -> str:
    """
    Return the path to the local media folder and ensure it exists.

    Defaults to `<cwd>/media` to match the ingestion pipeline behavior in v5.
    """
    media_path = os.path.join(base_dir or os.getcwd(), "media")
    os.makedirs(media_path, exist_ok=True)
    return media_path


async def get_file_hash(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute a stable SHA256 hash for the file at *path*."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()

