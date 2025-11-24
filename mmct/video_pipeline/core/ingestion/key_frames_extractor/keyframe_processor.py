"""
KeyframeProcessor: High-level orchestrator for keyframe processing pipeline.

Phase 1: Extracts keyframes and saves metadata to local JSON file.
Embeddings and search indexing happen in later phases.
"""

import os
import json
from typing import Optional
from loguru import logger

from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import (
    KeyframeExtractor,
    KeyframeExtractionConfig,
)
from mmct.video_pipeline.core.ingestion.models import (
    KeyframeMetadata,
    KeyframeMetadataCollection,
)
from mmct.video_pipeline.utils.helper import get_media_folder


class KeyframeProcessor:
    """
    Orchestrates keyframe extraction and local metadata storage:
    1. Extract keyframes from video
    2. Save metadata to local JSON file (without embeddings)
    """

    def __init__(
        self,
        keyframe_config: KeyframeExtractionConfig,
    ):
        """
        Initialize the keyframe processor.

        Args:
            keyframe_config: Configuration for keyframe extraction
        """
        self.keyframe_config = keyframe_config

    async def process_keyframes(
        self,
        video_path: str,
        video_hash_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
        offset_time: Optional[float] = None,
    ) -> str:
        """
        Process keyframes for a video part: extract and save metadata to JSON.

        Args:
            video_path: Path to the video file
            video_hash_id: Hash ID for this video part
            parent_id: Hash ID of the parent/original video
            parent_duration: Duration of the parent video in seconds
            video_duration: Duration of this video part in seconds
            offset_time: Time offset in seconds for Part B videos (None for Part A)

        Returns:
            str: Path to the saved keyframe metadata JSON file
        """
        try:
            # Step 1: Extract keyframes
            logger.info(f"Extracting keyframes for video {video_hash_id}...")
            keyframe_extractor = KeyframeExtractor(self.keyframe_config)
            keyframe_metadata_list = await keyframe_extractor.extract_keyframes(
                video_path=video_path, video_id=video_hash_id, offset_time=offset_time
            )
            logger.info(f"Successfully extracted {len(keyframe_metadata_list)} keyframes")

            # Step 2: Save keyframes metadata to JSON
            json_file_path = await self._save_keyframe_metadata_to_json(
                keyframe_metadata_list=keyframe_metadata_list,
                video_hash_id=video_hash_id,
                parent_id=parent_id,
                parent_duration=parent_duration,
                video_duration=video_duration,
            )

            logger.info(f"Successfully saved keyframe metadata to {json_file_path}")
            return json_file_path

        except Exception as e:
            logger.exception(f"Exception occurred during keyframe processing: {e}")
            raise

    async def _save_keyframe_metadata_to_json(
        self,
        keyframe_metadata_list: list,
        video_hash_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
    ) -> str:
        """
        Save keyframe metadata to a local JSON file.

        Args:
            keyframe_metadata_list: List of FrameMetadata objects from keyframe extraction
            video_hash_id: Hash ID for this video part
            parent_id: Hash ID of the parent/original video
            parent_duration: Duration of the parent video in seconds
            video_duration: Duration of this video part in seconds

        Returns:
            str: Path to the saved JSON file
        """
        # Get media folder
        media_folder = await get_media_folder()
        keyframes_dir = os.path.join(media_folder, "keyframes", video_hash_id)

        # Ensure directory exists
        os.makedirs(keyframes_dir, exist_ok=True)

        # Convert FrameMetadata objects to KeyframeMetadata model objects
        keyframe_metadata_objects = []
        for frame_meta in keyframe_metadata_list:
            keyframe_filename = f"{video_hash_id}_{frame_meta.frame_number}.jpg"
            file_path = os.path.join(keyframes_dir, keyframe_filename)

            keyframe_meta = KeyframeMetadata(
                keyframe_filename=keyframe_filename,
                timestamp_seconds=frame_meta.timestamp_seconds,
                file_path=file_path,
                motion_score=frame_meta.motion_score,
                embeddings=None,  # Will be populated in Phase 2
                blob_url="",  # Will be populated in Phase 3
            )
            keyframe_metadata_objects.append(keyframe_meta)

        # Create the collection
        collection = KeyframeMetadataCollection(
            video_id=video_hash_id,
            parent_id=parent_id,
            video_duration=video_duration,
            parent_duration=parent_duration,
            keyframes=keyframe_metadata_objects,
        )

        # Save to JSON file
        json_file_path = os.path.join(keyframes_dir, f"keyframe_metadata_{video_hash_id}.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(collection.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved keyframe metadata for {len(keyframe_metadata_objects)} frames to {json_file_path}")
        return json_file_path
