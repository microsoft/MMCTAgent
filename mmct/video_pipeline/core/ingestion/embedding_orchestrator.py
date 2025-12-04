"""
Embedding Orchestrator Module

Phase 2: Generates all embeddings in parallel for keyframes, chapters, and object collections.
Reads metadata JSON files created in Phase 1 and updates them with embeddings.
"""

import os
import json
import asyncio
from typing import List
from loguru import logger

from mmct.video_pipeline.core.ingestion.models import (
    KeyframeMetadataCollection,
    ChapterMetadataCollection,
    ObjectCollectionMetadata,
)
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.utils.error_handler import handle_exceptions, convert_exceptions, ProviderException


class EmbeddingOrchestrator:
    """
    Orchestrates parallel embedding generation for all content types.

    Reads JSON metadata files from Phase 1, generates embeddings in parallel,
    and updates the JSON files with generated embeddings.
    """

    def __init__(
        self,
        embedding_provider,
        image_embedding_provider
    ):
        """
        Initialize the embedding orchestrator.

        Args:
            embedding_provider: Provider for text embeddings
            image_embedding_provider: Provider for image embeddings (CLIP)
        """
        self.embedding_provider = embedding_provider
        self.image_embedding_provider = image_embedding_provider

    async def generate_all_embeddings(self, video_id: str) -> None:
        """
        Generate all embeddings in parallel for keyframes, chapters, and object collection.

        Args:
            video_id: Unique identifier for the video
        """
        logger.info(f"Starting parallel embedding generation for video {video_id}")

        # Run all embedding generation tasks in parallel
        await asyncio.gather(
            self._generate_keyframe_embeddings(video_id),
            self._generate_chapter_embeddings(video_id),
            self._generate_object_collection_embeddings(video_id),
        )

        logger.info(f"All embeddings generated successfully for video {video_id}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def _generate_keyframe_embeddings(self, video_id: str) -> None:
        """
        Generate CLIP embeddings for all keyframes.

        Args:
            video_id: Unique identifier for the video
        """
        try:
            # Read keyframe metadata JSON
            media_folder = await get_media_folder()
            keyframes_dir = os.path.join(media_folder, "keyframes", video_id)
            json_file_path = os.path.join(keyframes_dir, f"keyframe_metadata_{video_id}.json")

            if not os.path.exists(json_file_path):
                logger.warning(f"Keyframe metadata JSON not found: {json_file_path}")
                return

            # Load metadata
            with open(json_file_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            keyframe_collection = KeyframeMetadataCollection(**metadata_dict)
            logger.info(f"Generating CLIP embeddings for {len(keyframe_collection.keyframes)} keyframes...")

            # Collect all valid keyframe file paths
            valid_keyframes = []
            valid_file_paths = []

            for keyframe in keyframe_collection.keyframes:
                if not os.path.exists(keyframe.file_path):
                    logger.warning(f"Keyframe file not found: {keyframe.file_path}")
                    continue
                valid_keyframes.append(keyframe)
                valid_file_paths.append(keyframe.file_path)

            if not valid_file_paths:
                logger.warning("No valid keyframe files found")
                return

            # Generate embeddings in batches using the provider
            batch_size = self.image_embedding_provider.batch_size
            for i in range(0, len(valid_file_paths), batch_size):
                batch_paths = valid_file_paths[i:i + batch_size]
                batch_keyframes = valid_keyframes[i:i + batch_size]

                # Generate embeddings for this batch
                batch_embeddings = await self.image_embedding_provider.batch_image_embedding(batch_paths)

                # Update keyframe metadata with embeddings
                for keyframe, embedding in zip(batch_keyframes, batch_embeddings):
                    keyframe.embeddings = embedding

            logger.info(f"Successfully generated embeddings for {len(valid_keyframes)} keyframes")

            # Save updated metadata back to JSON
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(keyframe_collection.model_dump(), f, indent=2, ensure_ascii=False)

            logger.info(f"Updated keyframe metadata with embeddings: {json_file_path}")

        except Exception as e:
            logger.error(f"Failed to generate keyframe embeddings: {e}")
            raise

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def _generate_chapter_embeddings(self, video_id: str) -> None:
        """
        Generate text embeddings for all chapters.

        Args:
            video_id: Unique identifier for the video
        """
        try:
            # Read chapter metadata JSON
            media_folder = await get_media_folder()
            chapters_dir = os.path.join(media_folder, "chapters")
            json_file_path = os.path.join(chapters_dir, f"chapters_{video_id}.json")

            if not os.path.exists(json_file_path):
                logger.warning(f"Chapter metadata JSON not found: {json_file_path}")
                return

            # Load metadata
            with open(json_file_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            chapter_collection = ChapterMetadataCollection(**metadata_dict)
            logger.info(f"Generating text embeddings for {len(chapter_collection.chapters)} chapters...")

            # Generate embeddings for all chapters in parallel
            embedding_tasks = []
            for chapter in chapter_collection.chapters:
                # Create chapter content string for embedding
                chapter_content = self._create_chapter_content_string(chapter)
                embedding_tasks.append(self.embedding_provider.embedding(chapter_content))

            # Wait for all embeddings to complete
            embeddings = await asyncio.gather(*embedding_tasks)

            # Update chapters with embeddings
            for chapter, embedding in zip(chapter_collection.chapters, embeddings):
                chapter.embeddings = embedding

            logger.info(f"Successfully generated embeddings for {len(chapter_collection.chapters)} chapters")

            # Save updated metadata back to JSON
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(chapter_collection.model_dump(), f, indent=2, ensure_ascii=False)

            logger.info(f"Updated chapter metadata with embeddings: {json_file_path}")

        except Exception as e:
            logger.error(f"Failed to generate chapter embeddings: {e}")
            raise

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def _generate_object_collection_embeddings(self, video_id: str) -> None:
        """
        Generate text embeddings for video summary in object collection.

        Args:
            video_id: Unique identifier for the video
        """
        try:
            # Read object collection metadata JSON
            media_folder = await get_media_folder()
            object_collections_dir = os.path.join(media_folder, "object_collections")
            json_file_path = os.path.join(object_collections_dir, f"object_collection_{video_id}.json")

            if not os.path.exists(json_file_path):
                logger.warning(f"Object collection metadata JSON not found: {json_file_path}")
                return

            # Load metadata
            with open(json_file_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)

            object_collection_metadata = ObjectCollectionMetadata(**metadata_dict)

            if not object_collection_metadata.video_summary:
                logger.warning("No video summary found, skipping embedding generation")
                return

            logger.info("Generating text embedding for video summary...")

            # Generate embedding for video summary
            embedding = await self.embedding_provider.embedding(object_collection_metadata.video_summary)
            object_collection_metadata.embeddings = embedding

            logger.info("Successfully generated embedding for video summary")

            # Save updated metadata back to JSON
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(object_collection_metadata.model_dump(), f, indent=2, ensure_ascii=False)

            logger.info(f"Updated object collection metadata with embeddings: {json_file_path}")

        except Exception as e:
            logger.error(f"Failed to generate object collection embeddings: {e}")
            raise

    def _create_chapter_content_string(self, chapter) -> str:
        """
        Create a content string from chapter metadata for embedding generation.

        Args:
            chapter: ChapterMetadata object

        Returns:
            str: Combined content string for embedding
        """
        # Combine all text fields into a single string for embedding
        text = f"{chapter.detailed_summary} "

        if chapter.action_taken and chapter.action_taken.lower() != "none":
            text += f"The following actions are demonstrated in the video: {chapter.action_taken}. "

        if chapter.text_from_scene and chapter.text_from_scene.lower() != "none":
            text += f"Text visible in the video includes: {chapter.text_from_scene}. "

        # Add transcript
        if chapter.chapter_transcript:
            text += f"The complete transcript of the video is as follows: {chapter.chapter_transcript}"

        return text
