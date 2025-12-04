"""
Upload Orchestrator Module

Phase 3: Uploads all processed data to cloud storage and search indexes in parallel.
Reads metadata JSON files with embeddings from Phase 2 and performs bulk uploads.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from loguru import logger

from mmct.video_pipeline.core.ingestion.models import (
    KeyframeMetadataCollection,
    ChapterMetadataCollection,
    ObjectCollectionMetadata,
)
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_search_index import KeyframeSearchIndex
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.providers.search_document_models import ChapterIndexDocument
from mmct.utils.error_handler import handle_exceptions, convert_exceptions, ProviderException


class UploadOrchestrator:
    """
    Orchestrates parallel uploads of all processed data to storage and search indexes.

    Reads JSON metadata files with embeddings and uploads to:
    - Keyframe images to blob storage
    - Keyframe documents to search index
    - Chapter documents to search index
    - Object collection documents to search index
    """

    def __init__(self, search_provider:dict, blob_manager=None):
        """
        Initialize the upload orchestrator.

        Args:
            index_name: Base index name for search indexes
            search_provider: Search provider instance
            blob_manager: Optional blob storage manager
        """
        self.blob_manager = blob_manager
        self.search_provider_chapter = search_provider.get("chapter")
        self.search_provider_object_collection = search_provider.get("object_collection")
        self.search_provider_keyframe = search_provider.get("keyframe")

    async def upload_all(
        self,
        video_id: str,
        url: Optional[str] = None,
        keyframe_blob_url: Optional[str] = None,
    ) -> None:
        """
        Upload all processed data in parallel.

        Args:
            video_id: Unique identifier for the video
            url: Optional URL of the video
            keyframe_blob_url: Optional blob URL for keyframes folder
        """
        logger.info(f"Starting parallel uploads for video {video_id}")

        # Run all upload tasks in parallel
        await asyncio.gather(
            self._upload_keyframes(video_id),
            self._upload_chapters(video_id, url, keyframe_blob_url),
            self._upload_object_collection(video_id, url),
        )

        logger.info(f"All uploads completed successfully for video {video_id}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def _upload_keyframes(self, video_id: str) -> None:
        """
        Upload keyframe images to blob storage and index to search.

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
            logger.info(f"Uploading {len(keyframe_collection.keyframes)} keyframes...")

            # Upload keyframe images to blob storage if blob_manager is available
            if self.blob_manager:
                for keyframe in keyframe_collection.keyframes:
                    if not os.path.exists(keyframe.file_path):
                        logger.warning(f"Keyframe file not found: {keyframe.file_path}")
                        continue

                    # Upload to blob storage
                    blob_url = await self.blob_manager.upload_file(
                        file_name=f"{video_id}/{keyframe.keyframe_filename}",
                        src_file_path=keyframe.file_path,
                        folder_name=self.blob_manager.keyframe_container_name,
                    )

                    # Update blob_url in metadata
                    keyframe.blob_url = blob_url

                logger.info(f"Uploaded {len(keyframe_collection.keyframes)} keyframe images to blob storage")

            # Create search documents for keyframes
            documents = []
            for keyframe in keyframe_collection.keyframes:
                if not keyframe.embeddings:
                    logger.warning(f"Keyframe {keyframe.keyframe_filename} has no embeddings, skipping")
                    continue

                # Generate deterministic ID
                frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_id}_{keyframe.keyframe_filename}"))

                doc = {
                    "id": frame_id,
                    "video_id": keyframe_collection.video_id,
                    "keyframe_filename": keyframe.keyframe_filename,
                    "embeddings": keyframe.embeddings,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "motion_score": keyframe.motion_score,
                    "timestamp_seconds": keyframe.timestamp_seconds,
                    "blob_url": keyframe.blob_url or "",
                    "parent_id": keyframe_collection.parent_id,
                    "parent_duration": keyframe_collection.parent_duration,
                    "video_duration": keyframe_collection.video_duration,
                }
                documents.append(doc)

            # Upload to keyframe search index
            if documents:
                keyframe_search_index = KeyframeSearchIndex(
                    search_provider=self.search_provider_keyframe,
                )

                # Ensure index exists
                await keyframe_search_index.create_keyframe_index_if_not_exists()

                # Upload in batches
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    await self.search_provider_keyframe.upload_documents(batch)
                    logger.info(f"Uploaded batch {i // batch_size + 1} of {len(batch)} keyframe documents")

                logger.info(f"Successfully uploaded {len(documents)} keyframe documents to search index")

        except Exception as e:
            logger.error(f"Failed to upload keyframes: {e}")
            raise

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def _upload_chapters(
        self,
        video_id: str,
        url: Optional[str] = None,
        keyframe_blob_url: Optional[str] = None,
    ) -> None:
        """
        Upload chapter documents to search index.

        Args:
            video_id: Unique identifier for the video
            url: Optional URL of the video
            keyframe_blob_url: Optional blob URL for keyframes folder
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
            logger.info(f"Uploading {len(chapter_collection.chapters)} chapters...")

            # Create search documents for chapters
            documents = []
            current_time = datetime.now()

            for chapter in chapter_collection.chapters:
                if not chapter.embeddings:
                    logger.warning(f"Chapter has no embeddings, skipping")
                    continue

                # Create ChapterIndexDocument
                doc = ChapterIndexDocument(
                    id=str(uuid.uuid4()),
                    hash_video_id=chapter_collection.hash_video_id,
                    topic_of_video=chapter.topic_of_video,
                    action_taken=chapter.action_taken,
                    detailed_summary=chapter.detailed_summary,
                    category=chapter.category,
                    sub_category=chapter.sub_category,
                    text_from_scene=chapter.text_from_scene,
                    object_collection=chapter.object_collection,
                    url=chapter_collection.url,
                    time=current_time,
                    chapter_transcript=chapter.chapter_transcript,
                    parent_id=chapter_collection.parent_id,
                    parent_duration=chapter_collection.parent_duration,
                    video_duration=chapter_collection.video_duration,
                    start_time=chapter.start_time,
                    end_time=chapter.end_time,
                    blob_frames_folder_path=keyframe_blob_url or chapter.blob_frames_folder_path,
                    embeddings=chapter.embeddings,
                )
                documents.append(doc.model_dump())

            # Ensure chapter index exists
            index_exists = await self.search_provider_chapter.index_exists()
            if not index_exists:
                logger.info(f"Creating chapter index '{self.search_provider_chapter.index_name}'...")
                await self.search_provider_chapter.create_index()

            # Upload to search index
            if documents:
                await self.search_provider_chapter.upload_documents(
                    documents=documents,
                )
                logger.info(f"Successfully uploaded {len(documents)} chapter documents to search index")

        except Exception as e:
            logger.error(f"Failed to upload chapters: {e}")
            raise

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def _upload_object_collection(self, video_id: str, url: Optional[str] = None) -> None:
        """
        Upload object collection document to search index.

        Args:
            video_id: Unique identifier for the video
            url: Optional URL of the video
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
            logger.info("Uploading object collection...")

            if not object_collection_metadata.embeddings:
                logger.warning("Object collection has no embeddings, creating empty embedding")
                object_collection_metadata.embeddings = []

            # Create search document
            doc = {
                "id": object_collection_metadata.id,
                "video_id": object_collection_metadata.video_id,
                "url": object_collection_metadata.url,
                "object_collection": object_collection_metadata.object_collection,
                "object_count": object_collection_metadata.object_count,
                "video_summary": object_collection_metadata.video_summary,
                "embeddings": object_collection_metadata.embeddings,
                "video_duration": object_collection_metadata.video_duration,
            }

            # Ensure object collection index exists
            index_exists = await self.search_provider_object_collection.index_exists()
            if not index_exists:
                logger.info(f"Creating object collection index '{self.search_provider_object_collection.index_name}'...")
                await self.search_provider_object_collection.create_index()

            # Upload to search index
            await self.search_provider_object_collection.upload_documents(
                documents=[doc]
            )
            logger.info("Successfully uploaded object collection document to search index")

        except Exception as e:
            logger.error(f"Failed to upload object collection: {e}")
            raise
