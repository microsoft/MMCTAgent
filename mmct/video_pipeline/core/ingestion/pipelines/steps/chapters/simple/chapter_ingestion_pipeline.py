"""
Chapter Ingestion Pipeline Module

This module orchestrates the complete chapter generation and local storage workflow.
It coordinates semantic chunking, chapter generation, and saving to JSON files.
"""

import os
import json
from typing import List, Optional, Tuple, Any
from loguru import logger
from .chapter_generator import ChapterGenerator
from .object_collection_processor import ObjectCollectionProcessor
from mmct.video_pipeline.core.ingestion.models import ChapterMetadata, ChapterMetadataCollection
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.providers.base import BaseLLMProvider, BaseEmbeddingProvider

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)


class ChapterIngestionPipeline:
    """
    Chapter generation from chunks
    """

    def __init__(
        self,
        hash_id: str,
        keyframe_blob_url: str,
        llm_provider: BaseLLMProvider,
        embedding_provider: BaseEmbeddingProvider,
        frame_stacking_grid_size: int = 4,
        video_duration: Optional[float] = None,
        max_concurrent_requests: int = 3,
        max_chapter_duration: Optional[int] = None,
    ) -> None:
        """
        Initialize ChapterIngestionPipeline.

        Args:
            hash_id: Unique identifier for the video
            keyframe_blob_url: URL to keyframe blob storage folder
            llm_provider: LLM provider instance (handles both text and vision tasks)
            embedding_provider: Embedding provider instance (required)
            frame_stacking_grid_size: Grid size for frame stacking (default: 4)
            video_duration: Duration of current video
            max_concurrent_requests: Maximum concurrent LLM requests for chapter generation
            max_chapter_duration: Maximum duration (in seconds) of each chapter
        """
        # Core attributes
        self.hash_id = hash_id
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.video_duration = video_duration
        self.keyframe_blob_url = keyframe_blob_url

        # Store providers
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider

        # Initialize components with providers

        self.chapter_generator = ChapterGenerator(
            frame_stacking_grid_size=frame_stacking_grid_size,
            llm_provider=self.llm_provider,
            max_concurrent_requests=max_concurrent_requests,
        )

        # Initialize object collection processor
        self.object_collection_processor = ObjectCollectionProcessor(
            llm_provider=self.llm_provider,
        )

        # Pipeline state
        self.chunked_segments = []  # Store passed chunks
        self.chapter_responses = []
        self.chapter_transcripts = []
        self.chapter_timestamps = []

    async def run(
        self, url: Optional[str] = None, chunks: Optional[List[Any]] = None
    ) -> Tuple[Optional[List], Optional[List], str]:
        """
        Execute the complete chapter generation and local storage pipeline.

        Args:
            url: Optional YouTube URL for the video
            chunks: Pre-computed semantic chunks (optional if passed in init)

        Returns:
            Tuple of (chapter_responses, chapter_transcripts, chapters_json_path)
        """
        # Update chunks if passed in run
        if chunks:
            self.chunked_segments = chunks

        if not self.chunked_segments:
            logger.error("No semantic chunks provided to ChapterIngestionPipeline")
            return None, None, ""
        # Step 1: Generate chapters
        logger.info("Step 1: Generating chapters from semantic chunks...")
        await self._create_chapters()

        # Step 2: Process object collection (save to JSON)
        logger.info("Step 2: Processing object collection...")
        merged_registry, object_json_path = await self.object_collection_processor.run(
            chapter_responses=self.chapter_responses,
            video_id=self.hash_id,
            url=url,
            video_duration=self.video_duration,
        )
        if merged_registry:
            logger.info(
                f"Object collection processed: {len(merged_registry)} unique objects, saved to {object_json_path}"
            )
        else:
            logger.info(
                f"No objects found in chapters, saved empty collection to {object_json_path}"
            )

        # Step 3: Save chapters to JSON (without embeddings)
        logger.info("Step 3: Saving chapters to JSON...")
        chapters_json_path = await self._save_chapters_to_json(url=url)

        logger.info("Chapter pipeline completed successfully!")
        return self.chapter_responses, self.chapter_transcripts, chapters_json_path

    async def _create_chapters(self):
        """Create chapters using ChapterGenerator class."""
        if not self.chunked_segments:
            logger.warning("No chunked segments available for chapter creation")
            return

        # Use the chapter generator to create chapters in batch
        # Note: max_concurrent_requests is set in ChapterGenerator.__init__
        self.chapter_responses, self.chapter_transcripts, self.chapter_timestamps = (
            await self.chapter_generator.create_chapters_batch(
                chunked_segments=self.chunked_segments,
                video_id=self.hash_id,
                subject_variety={},
                categories="",
            )
        )

        logger.info(
            f"Chapter creation completed: {len(self.chapter_responses)} chapters created with timestamps"
        )

    async def _save_chapters_to_json(self, url: Optional[str] = None) -> str:
        """
        Create chapter metadata and save to local JSON file (without embeddings).

        Args:
            url: Optional YouTube URL for the video

        Returns:
            str: Path to the saved JSON file
        """
        chapter_metadata_list: List[ChapterMetadata] = []

        logger.info(f"Creating chapter metadata from {len(self.chapter_responses)} chapters")

        for chapter_response, chapter_transcript, timestamps in zip(
            self.chapter_responses, self.chapter_transcripts, self.chapter_timestamps
        ):
            # Serialize object_collection to JSON string
            object_collection_json = "[]"
            if chapter_response.object_collection:
                try:
                    # Convert the List[ObjectResponse] to JSON-serializable list
                    object_collection_list = [
                        obj.model_dump() for obj in chapter_response.object_collection
                    ]
                    object_collection_json = json.dumps(object_collection_list)
                except Exception as e:
                    logger.warning(f"Failed to serialize object_collection: {e}")
                    object_collection_json = "[]"

            # Extract start and end times from timestamps
            start_time = timestamps[0] if timestamps and len(timestamps) > 0 else 0.0
            end_time = timestamps[1] if timestamps and len(timestamps) > 1 else 0.0

            chapter_meta = ChapterMetadata(
                topic_of_video="None",
                action_taken=chapter_response.action_taken or "None",
                detailed_summary=chapter_response.detailed_summary or "None",
                category="None",
                sub_category="None",
                text_from_scene=chapter_response.text_from_scene or "None",
                object_collection=object_collection_json,
                chapter_transcript=chapter_transcript,
                blob_frames_folder_path=self.keyframe_blob_url or "None",
                start_time=start_time,
                end_time=end_time,
                embeddings=None,  # Will be populated in Phase 2
            )
            chapter_metadata_list.append(chapter_meta)

        # Create the collection
        collection = ChapterMetadataCollection(
            hash_video_id=self.hash_id,
            video_duration=str(self.video_duration) if self.video_duration is not None else "None",
            url=url or "None",
            chapters=chapter_metadata_list,
        )

        # Save to JSON file
        media_folder = await get_media_folder()
        chapters_dir = os.path.join(media_folder, "chapters")
        os.makedirs(chapters_dir, exist_ok=True)

        json_file_path = os.path.join(chapters_dir, f"chapters_{self.hash_id}.json")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(collection.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chapter_metadata_list)} chapters to {json_file_path}")
        return json_file_path
