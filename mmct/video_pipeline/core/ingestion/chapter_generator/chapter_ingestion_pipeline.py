"""
Chapter Ingestion Pipeline Module

This module orchestrates the complete chapter generation and ingestion workflow.
It coordinates semantic chunking, chapter generation, and search index ingestion.
"""

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
from mmct.video_pipeline.core.ingestion.chapter_generator.utils import create_embedding
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

    async def _ingest(self, url: Optional[str] = None):
        """
        Create search documents from chapters and ingest to search index.

        Args:
            url: Optional YouTube URL for the video
        """
        doc_objects: List[ChapterIndexDocument] = []
        current_time = datetime.now()

        logger.info(f"Creating documents from {len(self.chapter_responses)} chapters")

        for chapter_response, chapter_transcript, timestamps in zip(
            self.chapter_responses, self.chapter_transcripts, self.chapter_timestamps
        ):
            chapter_content_str = chapter_response.__str__(transcript=chapter_transcript)

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

            obj = ChapterIndexDocument(
                id=str(uuid.uuid4()),
                hash_video_id=self.hash_id,
                topic_of_video="None",
                action_taken=chapter_response.action_taken or "None",
                detailed_summary=chapter_response.detailed_summary or "None",
                category="None",
                sub_category="None",
                text_from_scene=chapter_response.text_from_scene or "None",
                object_collection=object_collection_json,
                youtube_url=url or "None",
                time=current_time,
                chapter_transcript=chapter_transcript,
                parent_id=self.parent_id or "None",
                parent_duration=str(self.parent_duration) if self.parent_duration is not None else "None",
                video_duration=str(self.video_duration) if self.video_duration is not None else "None",
                start_time=start_time,
                end_time=end_time,
                blob_audio_url="None",
                blob_video_url="None",
                blob_transcript_file_url="None",
                blob_frames_folder_path=self.keyframe_blob_url or "None",
                embeddings=await create_embedding(chapter_content_str),
            )
            doc_objects.append(obj)

        logger.info(f"Generated {len(doc_objects)} documents to upload")

        if not doc_objects:
            logger.error("No documents created - cannot upload to search index!")
            return

        await self.search_provider.upload_documents(
            documents=[doc.model_dump() for doc in doc_objects],
            index_name=self.index_name
        )
        logger.info(f"Successfully uploaded {len(doc_objects)} documents to index")

    async def run(self, url: Optional[str] = None) -> Tuple[Optional[List], Optional[List], bool]:
        """
        Execute the complete chapter ingestion pipeline.

        Args:
            url: Optional YouTube URL for the video

        Returns:
            Tuple of (chapter_responses, chapter_transcripts, is_already_ingested)
        """
        # Ensure search index exists before checking for duplicates
        await self._create_search_index()

        # Check for duplicate (only after ensuring index exists)
        is_exist = await self.search_provider.check_is_document_exist(
            hash_id=self.hash_id,
            index_name=self.index_name
        )
        if is_exist:
            logger.info("Document already exists in the index.")
            return None, None, is_exist

        # Step 1: Semantic Chunking
        logger.info("Step 1: Performing semantic chunking...")
        self.chunked_segments = await self.semantic_chunker.run()

        if not self.chunked_segments:
            logger.error("Semantic chunking failed - no segments created")
            return None, None, is_exist

        # Step 2: Generate chapters
        logger.info("Step 2: Generating chapters from semantic chunks...")
        await self._create_chapters()

        # Step 3: Process object collection and video summary
        logger.info("Step 3: Processing and indexing object collection and video summary...")
        merged_registry = await self.object_collection_processor.run(
            chapter_responses=self.chapter_responses,
            video_id=self.hash_id,
            url=url,
            video_duration=self.video_duration
        )
        if merged_registry:
            logger.info(f"Object collection processed: {len(merged_registry)} unique objects")
        else:
            logger.info("No objects found in chapters")

        # Step 4: Ingest to search index
        logger.info("Step 4: Ingesting chapters to search index...")
        await self._ingest(url=url)

        # Cleanup
        await self.search_provider.close()

        logger.info("Chapter ingestion pipeline completed successfully!")
        return self.chapter_responses, self.chapter_transcripts, is_exist


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
