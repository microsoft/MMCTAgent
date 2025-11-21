"""
Chapter Ingestion Pipeline Module

This module orchestrates the complete chapter generation and local storage workflow.
It coordinates semantic chunking, chapter generation, and saving to JSON files.
"""

import os
import uuid
import asyncio
import json
from datetime import datetime
from typing import List, Optional, Tuple
from loguru import logger

from mmct.providers.search_document_models import ChapterIndexDocument
from mmct.video_pipeline.core.ingestion.semantic_chunking.semantic_chunker import SemanticChunker
from mmct.video_pipeline.core.ingestion.chapter_generator.chapter_generator import ChapterGenerator
from mmct.video_pipeline.core.ingestion.chapter_generator.object_collection_processor import ObjectCollectionProcessor
from mmct.video_pipeline.core.ingestion.models import ChapterMetadata, ChapterMetadataCollection
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.providers.factory import provider_factory

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)


class ChapterIngestionPipeline:
    """
    Orchestrates the complete workflow of:
    1. Semantic chunking of transcripts
    2. Chapter generation from chunks
    3. Document creation and ingestion to search index

    """

    def __init__(
        self,
        hash_id: str,
        index_name: str,
        transcript: str,
        keyframe_blob_url: str,
        frame_stacking_grid_size: int = 4,
        parent_id: Optional[str] = None,
        parent_duration: Optional[float] = None,
        video_duration: Optional[float] = None,
    ) -> None:
        """
        Initialize ChapterIngestionPipeline.

        Args:
            hash_id: Unique identifier for the video
            index_name: Azure AI Search index name
            transcript: Raw SRT transcript text
            keyframe_blob_url: URL to keyframe blob storage folder
            frame_stacking_grid_size: Grid size for frame stacking (default: 4)
            parent_id: ID of parent video if this is a part
            parent_duration: Duration of parent video
            video_duration: Duration of current video
        """
        # Core attributes
        self.transcript = transcript
        self.hash_id = hash_id
        self.index_name = index_name
        self.frame_stacking_grid_size = frame_stacking_grid_size
        self.parent_id = parent_id
        self.parent_duration = parent_duration
        self.video_duration = video_duration
        self.keyframe_blob_url = keyframe_blob_url

        # Initialize components
        self.semantic_chunker = SemanticChunker(transcript=transcript)
        self.chapter_generator = ChapterGenerator(
            frame_stacking_grid_size=frame_stacking_grid_size,
            keyframe_index=f"keyframes-{index_name}",
        )

        # Initialize object collection processor
        self.object_collection_processor = ObjectCollectionProcessor(
            index_name=f"object-collection-{index_name}"
        )

        # Create search provider with custom index_name for this pipeline
        self.search_provider = provider_factory.create_search_provider()

        # Pipeline state
        self.chunked_segments = []
        self.chapter_responses = []
        self.chapter_transcripts = []
        self.chapter_timestamps = []

    async def _create_search_index(self):
        """Create search index if it doesn't exist."""
        # Check if index exists
        exists = await self.search_provider.index_exists(self.index_name)
        if exists:
            logger.info(f"Index {self.index_name} already exists.")
            return

        # Index doesn't exist, create it
        # Provider will handle schema creation based on type indicator
        logger.info(f"Creating index '{self.index_name}'...")
        
        created = await self.search_provider.create_index(self.index_name, "chapter")
        if created:
            logger.info(f"Index {self.index_name} created successfully.")



    async def _create_chapters(self):
        """Create chapters using ChapterGenerator class."""
        if not self.chunked_segments:
            logger.warning("No chunked segments available for chapter creation")
            return

        # Use the chapter generator to create chapters in batch
        # Note: max_concurrent_requests is set in ChapterGenerator.__init__
        self.chapter_responses, self.chapter_transcripts, self.chapter_timestamps = await self.chapter_generator.create_chapters_batch(
            chunked_segments=self.chunked_segments,
            video_id=self.hash_id,
            subject_variety={},
            categories="",
        )

        logger.info(f"Chapter creation completed: {len(self.chapter_responses)} chapters created with timestamps")

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
                    object_collection_list = [obj.model_dump() for obj in chapter_response.object_collection]
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
            parent_id=self.parent_id or "None",
            parent_duration=str(self.parent_duration) if self.parent_duration is not None else "None",
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

    async def run(self, url: Optional[str] = None) -> Tuple[Optional[List], Optional[List], str]:
        """
        Execute the complete chapter generation and local storage pipeline.

        Args:
            url: Optional YouTube URL for the video

        Returns:
            Tuple of (chapter_responses, chapter_transcripts, chapters_json_path)
        """
        # Step 1: Semantic Chunking
        logger.info("Step 1: Performing semantic chunking...")
        self.chunked_segments = await self.semantic_chunker.run()

        if not self.chunked_segments:
            logger.error("Semantic chunking failed - no segments created")
            return None, None, ""

        # Step 2: Generate chapters
        logger.info("Step 2: Generating chapters from semantic chunks...")
        await self._create_chapters()

        # Step 3: Process object collection (save to JSON)
        logger.info("Step 3: Processing object collection...")
        merged_registry, object_json_path = await self.object_collection_processor.run(
            chapter_responses=self.chapter_responses,
            video_id=self.hash_id,
            url=url,
            video_duration=self.video_duration
        )
        if merged_registry:
            logger.info(f"Object collection processed: {len(merged_registry)} unique objects, saved to {object_json_path}")
        else:
            logger.info(f"No objects found in chapters, saved empty collection to {object_json_path}")

        # Step 4: Save chapters to JSON (without embeddings)
        logger.info("Step 4: Saving chapters to JSON...")
        chapters_json_path = await self._save_chapters_to_json(url=url)

        logger.info("Chapter pipeline completed successfully!")
        return self.chapter_responses, self.chapter_transcripts, chapters_json_path


if __name__ == "__main__":
    # Example usage
    sample_transcript = """1
00:00:00,000 --> 00:00:05,000
This is a sample video transcript.

2
00:00:05,000 --> 00:00:10,000
It demonstrates the chapter ingestion pipeline."""

    pipeline = ChapterIngestionPipeline(
        hash_id="test-hash-123",
        index_name="test-index",
        transcript=sample_transcript,
        keyframe_blob_url="https://example.com/keyframes",
    )
    asyncio.run(pipeline.run())
