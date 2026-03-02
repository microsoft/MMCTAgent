"""Service layer for looking up and downloading transcripts.

Given a raw video_id, this module:
1. Normalises the video_id for blob path compatibility.
2. Constructs the expected blob path: <norm-id>/transcript.srt
3. Downloads the transcript content if the blob exists.
4. Returns the transcript text as a string.
"""

from fastapi import HTTPException

from app.utilities.blob_helpers import (
    normalize_video_id,
    blob_exists_cached,
    download_blob,
)

CONTAINER_NAME = "video-transcript-lively"


async def lookup_transcript_content(video_id: str) -> str:
    """Return the transcript content as a string, or raise 404 if not found.

    Args:
        video_id: Raw video identifier (pre-normalisation).

    Returns:
        Transcript content as a string (SRT format).

    Raises:
        HTTPException: 404 if the transcript blob does not exist.
    """
    norm_id = normalize_video_id(video_id)
    blob_name = f"{norm_id}/transcript.srt"

    if not await blob_exists_cached(CONTAINER_NAME, blob_name):
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found for video_id '{video_id}' "
                   f"(normalized: '{norm_id}')",
        )

    data = await download_blob(CONTAINER_NAME, blob_name)
    if data is None:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to download transcript for video_id '{video_id}'",
        )

    return data.decode("utf-8")
