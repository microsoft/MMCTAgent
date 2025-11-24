"""
Cleanup Manager Module

Phase 4: Cleans up temporary files and resources after successful upload.
Removes JSON metadata files and optionally keyframe images.
"""

import os
import shutil
from typing import Optional
from loguru import logger

from mmct.video_pipeline.utils.helper import get_media_folder, remove_file


class CleanupManager:
    """
    Manages cleanup of temporary files created during the ingestion pipeline.

    Removes metadata JSON files and optionally keyframe images after successful upload.
    """

    def __init__(self, keep_keyframes: bool = False):
        """
        Initialize the cleanup manager.

        Args:
            keep_keyframes: If True, keep keyframe image files (default: False)
        """
        self.keep_keyframes = keep_keyframes

    async def cleanup(self, video_id: str) -> None:
        """
        Clean up temporary files for a video.

        Args:
            video_id: Unique identifier for the video
        """
        logger.info(f"Starting cleanup for video {video_id}")

        media_folder = await get_media_folder()

        # Delete metadata JSON files
        await self._delete_file(os.path.join(media_folder, "keyframes", video_id, f"keyframe_metadata_{video_id}.json"))
        await self._delete_file(os.path.join(media_folder, "chapters", f"chapters_{video_id}.json"))
        await self._delete_file(os.path.join(media_folder, "object_collections", f"object_collection_{video_id}.json"))
        await self._delete_file(os.path.join(media_folder, f"transcript_{video_id}.srt"))

        # Delete audio files (created during transcription - both .wav and .mp3)
        await self._delete_file(os.path.join(media_folder, f"{video_id}.wav"))  # CloudTranscription/Azure STT
        await self._delete_file(os.path.join(media_folder, f"{video_id}.mp3"))  # WhisperTranscription

        # Delete copied video file (video renamed to hash_id during processing)
        await self._delete_file(os.path.join(media_folder, f"{video_id}.mp4"))

        # Delete keyframe images and directories using helper function
        if not self.keep_keyframes:
            await remove_file(video_id)

        logger.info(f"Cleanup completed for video {video_id}")

    async def _delete_file(self, file_path: str) -> None:
        """
        Delete a single file.

        Args:
            file_path: Full path to the file to delete
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            else:
                logger.debug(f"File not found: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
