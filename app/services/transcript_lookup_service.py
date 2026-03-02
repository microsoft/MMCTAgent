"""Service layer for looking up transcript blob URLs.

Given a raw video_id, this module:
1. Normalises the video_id for blob path compatibility.
2. Constructs the expected blob path: <norm-id>/transcript.srt
3. Checks whether the blob actually exists (TTL-cached).
4. Raises an error if the transcript is not found.
"""

from fastapi import HTTPException

from app.utilities.blob_helpers import (
    normalize_video_id,
    get_account_url,
    blob_exists_cached,
)

CONTAINER_NAME = "video-transcript-lively"


async def lookup_transcript_url(video_id: str) -> str:
    """Return the transcript blob URL, or raise 404 if it does not exist.

    Args:
        video_id: Raw video identifier (pre-normalisation).

    Returns:
        Full blob URL string.

    Raises:
        HTTPException: 404 if the transcript blob does not exist.
    """
    account_url = get_account_url()
    norm_id = normalize_video_id(video_id)
    blob_name = f"{norm_id}/transcript.srt"

    if not await blob_exists_cached(CONTAINER_NAME, blob_name):
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found for video_id '{video_id}' "
                   f"(normalized: '{norm_id}')",
        )

    return f"{account_url}/{CONTAINER_NAME}/{blob_name}"
