"""Service layer for transcript lookup.

Resolves a video_id to its stored transcript file and returns the content
as a UTF-8 string.

Flow:
  1. Normalise the video_id for blob path compatibility.
  2. Build the expected blob path: <normalised-id>/transcript.srt
  3. Fetch the file from blob storage via the configured storage provider.
  4. Return the decoded transcript text, or raise an HTTP error on failure.

The transcript container name is read from application settings so that no
storage path is hardcoded in this file.
"""

from fastapi import HTTPException
from loguru import logger

from app.config import get_settings, get_video_agent_provider
from app.utilities.blob_helpers import normalize_video_id


async def lookup_transcript_content(video_id: str) -> str:
    """Return the transcript text for a video, or raise an HTTP exception.

    Fetches the SRT transcript file from blob storage using the configured
    storage provider. The container name is read from settings so callers
    are not coupled to a specific storage path.

    Args:
        video_id: Raw video identifier (will be normalised before use).

    Returns:
        Full transcript content as a UTF-8 string (SRT format).

    Raises:
        HTTPException 404: Transcript blob does not exist in storage.
        HTTPException 502: Blob was found but could not be downloaded.
        HTTPException 500: Storage provider initialisation or unexpected failure.
    """
    settings = get_settings()
    container = settings.transcript_container_name
    norm_id = normalize_video_id(video_id)
    blob_path = f"{norm_id}/transcript.srt"

    logger.debug(f"Fetching transcript: container={container}, blob={blob_path}")

    try:
        storage_provider = get_video_agent_provider().storage_provider
        data: bytes = await storage_provider.load_file_to_memory(
            folder=container,
            file_name=blob_path,
        )
    except Exception as exc:
        error_msg = str(exc).lower()
        # Treat blob-not-found errors as 404
        if any(k in error_msg for k in ("blobnotfound", "does not exist", "404", "not found")):
            raise HTTPException(
                status_code=404,
                detail=f"Transcript not found for video_id '{video_id}' "
                       f"(normalised: '{norm_id}')",
            )
        logger.error(f"Transcript fetch failed for video_id='{video_id}': {exc}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to retrieve transcript for video_id '{video_id}'",
        )

    if not data:
        raise HTTPException(
            status_code=502,
            detail=f"Empty transcript returned for video_id '{video_id}'",
        )

    return data.decode("utf-8")
