"""
Chapter Ingestion Pipeline Module

This module orchestrates the complete chapter generation and ingestion workflow.
It coordinates semantic chunking, chapter generation, and search index ingestion.
"""

import uuid
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from loguru import logger

from mmct.providers.search_document_models import ChapterIndexDocument
from mmct.video_pipeline.core.ingestion.semantic_chunking.semantic_chunker import SemanticChunker
from mmct.video_pipeline.core.ingestion.chapter_generator.chapter_generator import ChapterGenerator
from mmct.video_pipeline.core.ingestion.chapter_generator.subject_registry_processor import SubjectRegistryProcessor
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
        self.embedding_provider = provider_factory.create_embedding_provider()

        # Initialize subject registry processor
        self.subject_registry_processor = SubjectRegistryProcessor(
            index_name=f"subject-registry-{index_name}"
        )

        # Create search provider with custom index_name for this pipeline
        self.search_provider = provider_factory.create_search_provider()

        # Pipeline state
        self.chunked_segments = []
        self.chapter_responses = []
        self.chapter_transcripts = []

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

    async def _create_embedding_normal(self, text: str) -> List[float]:
        """Create embedding for text."""
        try:
            return await self.embedding_provider.embedding(text)
        except Exception as e:
            raise Exception(f"Failed to create embedding: {e}")


    async def _create_chapters(self):
        """Create chapters using ChapterGenerator class."""
        if not self.chunked_segments:
            logger.warning("No chunked segments available for chapter creation")
            return

        # Use the chapter generator to create chapters in batch
        # Note: max_concurrent_requests is set in ChapterGenerator.__init__
        self.chapter_responses, self.chapter_transcripts = await self.chapter_generator.create_chapters_batch(
            chunked_segments=self.chunked_segments,
            video_id=self.hash_id,
            subject_variety={},
            categories="",
        )

        logger.info(f"Chapter creation completed: {len(self.chapter_responses)} chapters created")

    async def _ingest(self, url: Optional[str] = None):
        """
        Create search documents from chapters and ingest to search index.

        Args:
            url: Optional YouTube URL for the video
        """
        doc_objects: List[ChapterIndexDocument] = []
        current_time = datetime.now()

        logger.info(f"Creating documents from {len(self.chapter_responses)} chapters")

        for chapter_response, chapter_transcript in zip(
            self.chapter_responses, self.chapter_transcripts
        ):
            chapter_content_str = chapter_response.__str__(transcript=chapter_transcript)

            # Serialize subject_registry to JSON string
            subject_registry_json = "[]"
            if chapter_response.subject_registry:
                try:
                    # Convert the List[SubjectResponse] to JSON-serializable list
                    subject_registry_list = [subject.model_dump() for subject in chapter_response.subject_registry]
                    subject_registry_json = json.dumps(subject_registry_list)
                except Exception as e:
                    logger.warning(f"Failed to serialize subject_registry: {e}")
                    subject_registry_json = "[]"

            obj = ChapterIndexDocument(
                id=str(uuid.uuid4()),
                hash_video_id=self.hash_id,
                topic_of_video=chapter_response.topic_of_video or "None",
                action_taken=chapter_response.action_taken or "None",
                detailed_summary=chapter_response.detailed_summary or "None",
                category=chapter_response.category or "None",
                sub_category=chapter_response.sub_category or "None",
                text_from_scene=chapter_response.text_from_scene or "None",
                subject_registry=subject_registry_json,
                youtube_url=url or "None",
                time=current_time,
                chapter_transcript=chapter_transcript,
                parent_id=self.parent_id or "None",
                parent_duration=str(self.parent_duration) if self.parent_duration is not None else "None",
                video_duration=str(self.video_duration) if self.video_duration is not None else "None",
                blob_audio_url="None",
                blob_video_url="None",
                blob_transcript_file_url="None",
                blob_frames_folder_path=self.keyframe_blob_url or "None",
                embeddings=await self._create_embedding_normal(chapter_content_str),
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

        # Step 3: Process subject registry
        logger.info("Step 3: Processing and indexing subject registry...")
        merged_registry = await self.subject_registry_processor.run(
            chapter_responses=self.chapter_responses,
            video_id=self.hash_id,
            url=url
        )
        if merged_registry:
            logger.info(f"Subject registry processed: {len(merged_registry)} unique subjects")
        else:
            logger.info("No subjects found in chapters")

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
