"""
Azure AI Search index management for keyframe embeddings.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
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
from ..core import FrameEmbedding, SearchIndexConfig
from ..auth import get_azure_credential

logger = logging.getLogger(__name__)


class KeyframeSearchIndex:
    """Manages keyframe storage and search in Azure AI Search."""

    def __init__(
        self,
        search_endpoint: str,
        index_name: str = "video-keyframes-index",
        api_key: Optional[str] = None,
        config: Optional[SearchIndexConfig] = None
    ):
        """
        Initialize the keyframe search index.

        Authentication priority:
        1. Azure CLI credentials
        2. DefaultAzureCredential (Managed Identity, Environment, etc.)
        3. API key (if provided)

        Args:
            search_endpoint: Azure AI Search endpoint URL
            index_name: Name of the search index for keyframes
            api_key: Optional API key for authentication (fallback)
            config: Optional SearchIndexConfig
        """
        self.search_endpoint = search_endpoint
        self.index_name = index_name
        self.config = config or SearchIndexConfig()

        # Setup authentication with priority order
        logger.info("Initializing Azure AI Search authentication")
        self.credential = get_azure_credential(api_key)

        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=search_endpoint,
            credential=self.credential
        )
        self.search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=self.credential
        )


    async def create_keyframe_index_if_not_exists(self) -> bool:
        """
        Create the keyframe index with the proper schema if it doesn't exist.

        Returns:
            bool: True if index was created, False if it already existed
        """
        try:
            # Check if index exists
            try:
                existing_index = self.index_client.get_index(self.index_name)
                if existing_index:
                    logger.info(f"Keyframe index '{self.index_name}' already exists")
                    return False
            except Exception as e:
                if "ResourceNotFound" in str(e) or "NotFound" in str(e) or "does not exist" in str(e):
                    logger.info(f"Keyframe index '{self.index_name}' does not exist, will create")
                else:
                    logger.error(f"Error checking if index exists: {e}")

            # Define keyframe-specific fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(
                    name="video_id",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SearchableField(
                    name="keyframe_filename",
                    type=SearchFieldDataType.String,
                    filterable=True,
                    facetable=True
                ),
                SearchField(
                    name="clip_embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    retrievable=True,
                    vector_search_dimensions=self.config.vector_dimensions,
                    vector_search_profile_name="clip-profile"
                ),
                SimpleField(
                    name="created_at",
                    type=SearchFieldDataType.DateTimeOffset,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="motion_score",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="timestamp_seconds",
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True
                ),
                SimpleField(
                    name="blob_url",
                    type=SearchFieldDataType.String,
                    retrievable=True
                ),
                SimpleField(
                    name="youtube_url",
                    type=SearchFieldDataType.String,
                    retrievable=True,
                    filterable=True
                )
            ]

            # Configure vector search for CLIP embeddings
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-algorithm",
                        parameters={
                            "m": self.config.hnsw_m,
                            "efConstruction": self.config.hnsw_ef_construction,
                            "efSearch": self.config.hnsw_ef_search,
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

            try:
                self.index_client.create_index(index)
                logger.info(f"Successfully created keyframe index '{self.index_name}'")
            except Exception as create_error:
                if "ResourceNameAlreadyInUse" in str(create_error) or "already exists" in str(create_error):
                    logger.info(f"Keyframe index '{self.index_name}' already exists (detected during creation)")
                    return False
                else:
                    raise

            return True

        except Exception as e:
            logger.error(f"Failed to create keyframe index: {e}")
            raise

    def create_frame_documents(
        self,
        frame_embeddings: List[FrameEmbedding],
        video_id: str,
        video_path: str,
        youtube_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create search documents from frame embeddings.

        Args:
            frame_embeddings: List of FrameEmbedding objects
            video_id: Unique video identifier
            video_path: Path to the video file
            youtube_url: Optional YouTube URL for the video

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
                "blob_url": frame_embedding.blob_url or "",
                "youtube_url": youtube_url or ""
            }

            documents.append(document)

        return documents

    async def upload_frame_embeddings(
        self,
        frame_embeddings: List[FrameEmbedding],
        video_id: str,
        video_path: str,
        youtube_url: Optional[str] = None
    ) -> bool:
        """
        Upload frame embeddings to the search index.

        Args:
            frame_embeddings: List of FrameEmbedding objects
            video_id: Unique video identifier
            video_path: Path to the video file
            youtube_url: Optional YouTube URL for the video

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not frame_embeddings:
                logger.warning("No frame embeddings to upload")
                return True

            # Ensure index exists
            try:
                await self.create_keyframe_index_if_not_exists()
            except Exception as index_error:
                logger.warning(f"Index creation check failed: {index_error}")

            # Create documents
            documents = self.create_frame_documents(frame_embeddings, video_id, video_path, youtube_url)

            # Upload in batches
            batch_size = self.config.batch_upload_size
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = self.search_client.upload_documents(documents=batch)
                logger.info(f"Uploaded batch {i // batch_size + 1} of {len(batch)} frame documents")

            logger.info(f"Successfully uploaded {len(documents)} frame embeddings to search index")
            return True

        except Exception as e:
            logger.error(f"Failed to upload frame embeddings: {e}")
            return False

    async def search_similar_frames(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar frames using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            video_id: Optional video ID to filter results

        Returns:
            List of search results
        """
        try:
            from azure.search.documents.models import VectorizedQuery

            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="clip_embedding"
            )

            search_params = {
                "vector_queries": [vector_query],
                "top": top_k
            }

            if video_id:
                search_params["filter"] = f"video_id eq '{video_id}'"

            results = self.search_client.search(**search_params)

            return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Failed to search similar frames: {e}")
            return []

    async def close(self):
        """Close the search client."""
        try:
            await self.search_client.close()
            await self.index_client.close()
        except Exception as e:
            logger.warning(f"Error closing search clients: {e}")
