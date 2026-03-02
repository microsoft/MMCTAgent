"""Service layer for looking up uniform-frame blob URLs.

Given a raw video_id and a timestamp (seconds), this module:
1. Normalises the video_id for blob path compatibility.
2. Constructs blob paths for t-1, t, t+1.
3. Checks whether each blob actually exists (TTL-cached).
4. Returns only the URLs that resolve to real blobs.
"""

import asyncio
from typing import List, Tuple

from app.utilities.blob_helpers import (
    normalize_video_id,
    get_account_url,
    blob_exists_cached,
)

CONTAINER_NAME = "video-frames-lively"
EXTENSION = "jpg"


async def lookup_frame_urls(
    video_id: str,
    timestamp: int,
) -> List[Tuple[int, str]]:
    """Return list of (timestamp_second, blob_url) for existing frames in ±1s.

    Args:
        video_id: Raw video identifier (pre-normalisation).
        timestamp: Centre timestamp in seconds.

    Returns:
        List of (ts, url) tuples for blobs that exist.
    """
    account_url = get_account_url()
    norm_id = normalize_video_id(video_id)
    candidates = [timestamp - 1, timestamp, timestamp + 1]

    results: List[Tuple[int, str]] = []

    async def _check(ts: int) -> None:
        blob_name = f"{norm_id}/{ts}/frame.{EXTENSION}"
        if await blob_exists_cached(CONTAINER_NAME, blob_name):
            url = f"{account_url}/{CONTAINER_NAME}/{blob_name}"
            results.append((ts, url))

    await asyncio.gather(*[_check(ts) for ts in candidates])

    results.sort(key=lambda t: t[0])
    return results
