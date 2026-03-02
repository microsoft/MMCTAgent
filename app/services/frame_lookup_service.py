"""Service layer for looking up and downloading uniform frames.

Given a raw video_id and a timestamp (seconds), this module:
1. Normalises the video_id for blob path compatibility.
2. Constructs blob paths for t-1, t, t+1.
3. Downloads frame bytes for each blob that exists (TTL-cached existence check).
4. Returns the raw image bytes for each frame.
"""

import asyncio
import base64
from typing import List, Tuple, Optional

from app.utilities.blob_helpers import (
    normalize_video_id,
    blob_exists_cached,
    download_blob,
)

CONTAINER_NAME = "video-frames-lively"
EXTENSION = "jpg"


async def lookup_frame_bytes(
    video_id: str,
    timestamp: int,
) -> List[Tuple[int, str]]:
    """Return list of (timestamp_second, base64_image) for existing frames in ±1s.

    Args:
        video_id: Raw video identifier (pre-normalisation).
        timestamp: Centre timestamp in seconds.

    Returns:
        List of (ts, base64_encoded_bytes) tuples for blobs that exist.
    """
    norm_id = normalize_video_id(video_id)
    candidates = [timestamp]

    results: List[Tuple[int, str]] = []

    async def _fetch(ts: int) -> None:
        blob_name = f"{norm_id}/{ts}/frame.{EXTENSION}"
        if await blob_exists_cached(CONTAINER_NAME, blob_name):
            data = await download_blob(CONTAINER_NAME, blob_name)
            if data is not None:
                results.append((ts, base64.b64encode(data).decode("ascii")))

    await asyncio.gather(*[_fetch(ts) for ts in candidates])

    results.sort(key=lambda t: t[0])
    return results
