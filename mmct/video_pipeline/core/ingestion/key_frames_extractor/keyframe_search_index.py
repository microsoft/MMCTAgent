import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import numpy as np
from mmct.video_pipeline.utils.ai_search_client import AISearchClient
from mmct.video_pipeline.core.ingestion.key_frames_extractor.clip_embeddings import FrameEmbedding
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
import logging

logger = logging.getLogger(__name__)

class KeyframeSearchIndex:
    """Manages keyframe storage and search in Azure AI Search."""

    def __init__(self, search_endpoint: str, index_name: str = "video-keyframes-index"):
        """
        Initialize the keyframe search index.

        Args:
            search_endpoint: Azure AI Search endpoint URL
            index_name: Name of the search index for keyframes
        """
        self.search_endpoint = search_endpoint
        self.index_name = index_name
        self.search_client = AISearchClient(
            endpoint=search_endpoint,
            index_name=index_name
        )

    async def create_keyframe_index_if_not_exists(self) -> bool:
        """
        Create the keyframe index with the proper schema if it doesn't exist.

        Returns:
            bool: True if index was created, False if it already existed
        """
        try:
            # Check if index exists with authentication retry
            try:
                async def _check_index():
                    return await self.search_client.index_client.get_index(self.index_name)

                existing_index = await self.search_client._retry_with_cli_credential(_check_index)
                if existing_index:
                    logger.info(f"Keyframe index '{self.index_name}' already exists")
                    return False
            except Exception as e:
                # Check if it's specifically a "not found" error vs other errors
                if "ResourceNotFound" in str(e) or "NotFound" in str(e) or "does not exist" in str(e):
                    logger.info(f"Keyframe index '{self.index_name}' does not exist, will create")
                else:
                    logger.error(f"Error checking if index exists: {e}")
                    # If we can't determine if index exists, try to create and handle the error below
                    pass

            # Define keyframe-specific fields (matching reference schema)
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="video_id", type=SearchFieldDataType.String,
                              filterable=True, facetable=True),
                SearchableField(name="keyframe_filename", type=SearchFieldDataType.String,
                              filterable=True, facetable=True),
                SearchField(name="clip_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                           searchable=True, retrievable=True, vector_search_dimensions=512,
                           vector_search_profile_name="clip-profile"),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset,
                           filterable=True, sortable=True),
                SimpleField(name="motion_score", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="timestamp_seconds", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="blob_url", type=SearchFieldDataType.String,
                           retrievable=True),
                SimpleField(name="frame_path", type=SearchFieldDataType.String,
                           retrievable=True)
            ]

            # Configure vector search for CLIP embeddings
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-algorithm",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="clip-profile",
                        algorithm_configuration_name="hnsw-algorithm"
                    )
                ]
            )

            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )

            # Use retry mechanism for authentication fallback
            async def _create_index():
                return await self.search_client.index_client.create_index(index)

            try:
                await self.search_client._retry_with_cli_credential(_create_index)
                logger.info(f"Successfully created keyframe index '{self.index_name}'")
            except Exception as create_error:
                if "ResourceNameAlreadyInUse" in str(create_error) or "already exists" in str(create_error):
                    logger.info(f"Keyframe index '{self.index_name}' already exists (detected during creation)")
                    return False  # Index exists, that's fine
                else:
                    raise  # Re-raise other errors

            return True

        except Exception as e:
            logger.error(f"Failed to create keyframe index: {e}")
            raise

    def create_frame_documents(self, frame_embeddings: List[FrameEmbedding],
                             video_id: str, video_path: str) -> List[Dict[str, Any]]:
        """
        Create search documents from frame embeddings.

        Args:
            frame_embeddings: List of FrameEmbedding objects
            video_id: Unique video identifier
            video_path: Path to the video file

        Returns:
            List of document dictionaries ready for Azure AI Search
        """
        documents = []

        for frame_embedding in frame_embeddings:
            # Generate unique ID for this frame
            frame_id = f"{video_id}_{frame_embedding.frame_metadata.frame_number}"
            frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, frame_id))

            # Create frame filename
            frame_filename = f"{video_id}_{frame_embedding.frame_metadata.frame_number}.jpg"

            document = {
                "id": frame_id,
                "video_id": video_id,
                "keyframe_filename": frame_filename,
                "clip_embedding": frame_embedding.clip_embedding.tolist(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "motion_score": float(frame_embedding.frame_metadata.motion_score),
                "timestamp_seconds": float(frame_embedding.frame_metadata.timestamp_seconds),
                "blob_url": "",  # TODO: Add blob URL if needed
                "frame_path": frame_embedding.frame_path
            }

            documents.append(document)

        return documents

    async def upload_frame_embeddings(self, frame_embeddings: List[FrameEmbedding],
                                    video_id: str, video_path: str) -> bool:
        """
        Upload frame embeddings to the search index.

        Args:
            frame_embeddings: List of FrameEmbedding objects
            video_id: Unique video identifier
            video_path: Path to the video file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not frame_embeddings:
                logger.warning("No frame embeddings to upload")
                return True

            # Ensure index exists (or confirm it already exists)
            try:
                await self.create_keyframe_index_if_not_exists()
            except Exception as index_error:
                logger.warning(f"Index creation check failed: {index_error}")
                # Continue with upload anyway - index might already exist

            # Create documents
            documents = self.create_frame_documents(frame_embeddings, video_id, video_path)

            # Upload in batches with authentication retry
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                async def _upload_batch():
                    return await self.search_client.upload_documents(batch)

                result = await self.search_client._retry_with_cli_credential(_upload_batch)
                logger.info(f"Uploaded batch {i // batch_size + 1} of {len(batch)} frame documents")

            logger.info(f"Successfully uploaded {len(documents)} frame embeddings to search index")
            return True

        except Exception as e:
            logger.error(f"Failed to upload frame embeddings: {e}")
            return False

    async def close(self):
        """Close the search client."""
        await self.search_client.close()