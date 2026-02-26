"""
Cleanup Manager Module

Cleans up temporary files and resources after successful pipeline completion.
Removes all temporary files created during the temporal_graph_ingestion pipeline.

Files cleaned up:
- Compressed video: media/compressed/{video_id}.mp4
- Audio files: media/{video_id}.wav, media/{video_id}.mp3
- Dense keyframes: {output_dir}/dense_keyframes/{chunk_id}/*.jpg
- Transcript files: media/transcript_{video_id}.srt
- Legacy metadata JSON files (if present)
"""

import os
import shutil
from typing import Optional
from loguru import logger

from mmct.video_pipeline.utils.helper import get_media_folder


class CleanupManager:
    """
    Manages cleanup of temporary files created during the ingestion pipeline.

    Removes temporary files after successful upload/export.
    """

    def __init__(
        self,
        keep_keyframes: bool = False,
        keep_compressed: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the cleanup manager.

        Args:
            keep_keyframes: If True, keep keyframe image files (default: False)
            keep_compressed: If True, keep compressed video file (default: False)
            output_dir: Pipeline output directory containing dense_keyframes folder
        """
        self.keep_keyframes = keep_keyframes
        self.keep_compressed = keep_compressed
        self.output_dir = output_dir

    async def cleanup(self, video_id: str) -> int:
        """
        Clean up temporary files for a video.

        Args:
            video_id: Unique identifier for the video

        Returns:
            Number of items (files/directories) deleted
        """
        logger.debug(f"Starting cleanup for video {video_id}")
        deleted_count = 0

        media_folder = await get_media_folder()

        # ============================================================
        # 1. Clean up compressed video
        # ============================================================
        if not self.keep_compressed:
            compressed_path = os.path.join(media_folder, "compressed", f"{video_id}.mp4")
            if await self._delete_file(compressed_path):
                deleted_count += 1

        # ============================================================
        # 2. Clean up audio files (created during transcription)
        # ============================================================
        # Azure Speech STT creates .wav files
        if await self._delete_file(os.path.join(media_folder, f"{video_id}.wav")):
            deleted_count += 1
        # Whisper transcription may create .mp3 files
        if await self._delete_file(os.path.join(media_folder, f"{video_id}.mp3")):
            deleted_count += 1

        # ============================================================
        # 3. Clean up transcript files
        # ============================================================
        if await self._delete_file(os.path.join(media_folder, f"transcript_{video_id}.srt")):
            deleted_count += 1

        # ============================================================
        # 4. Clean up dense keyframes directory
        # ============================================================
        # Dense keyframes are stored in {output_dir}/dense_keyframes/{chunk_id}/
        if not self.keep_keyframes and self.output_dir:
            dense_keyframes_dir = os.path.join(self.output_dir, "dense_keyframes")
            if await self._delete_directory(dense_keyframes_dir):
                deleted_count += 1

        # Also clean up legacy keyframes location (media/keyframes/{video_id}/)
        if not self.keep_keyframes:
            legacy_keyframes_dir = os.path.join(media_folder, "keyframes", video_id)
            if await self._delete_directory(legacy_keyframes_dir):
                deleted_count += 1

        # ============================================================
        # 5. Clean up legacy metadata JSON files (older pipeline steps)
        # ============================================================
        legacy_files = [
            os.path.join(media_folder, "keyframes", video_id, f"keyframe_metadata_{video_id}.json"),
            os.path.join(media_folder, "chapters", f"chapters_{video_id}.json"),
            os.path.join(media_folder, "object_collections", f"object_collection_{video_id}.json"),
        ]
        for file_path in legacy_files:
            if await self._delete_file(file_path):
                deleted_count += 1

        # ============================================================
        # 6. Clean up copied video file (video renamed to hash_id during processing)
        # ============================================================
        if await self._delete_file(os.path.join(media_folder, f"{video_id}.mp4")):
            deleted_count += 1

        logger.debug(f"Cleanup completed for video {video_id}: {deleted_count} items deleted")
        return deleted_count

    async def _delete_file(self, file_path: str) -> bool:
        """
        Delete a single file.

        Args:
            file_path: Full path to the file to delete

        Returns:
            True if file was deleted, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file: {file_path}")
                return True
            else:
                logger.debug(f"File not found (skipping): {file_path}")
                return False
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")
            return False

    async def _delete_directory(self, dir_path: str) -> bool:
        """
        Delete a directory and all its contents.

        Args:
            dir_path: Full path to the directory to delete

        Returns:
            True if directory was deleted, False otherwise
        """
        try:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                logger.debug(f"Deleted directory: {dir_path}")
                return True
            else:
                logger.debug(f"Directory not found (skipping): {dir_path}")
                return False
        except Exception as e:
            logger.warning(f"Failed to delete directory {dir_path}: {e}")
            return False
