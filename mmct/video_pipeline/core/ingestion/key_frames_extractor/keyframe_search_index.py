from typing import List, Optional
import logging

from mmct.providers.azure_providers.search_provider import AzureSearchProvider
from mmct.video_pipeline.core.ingestion.key_frames_extractor.clip_embeddings import FrameEmbedding
from mmct.video_pipeline.utils.video_frame_search import create_frame_documents_from_embeddings
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

logger = logging.getLogger(__name__)


class KeyframeSearchIndex:
    """Manages keyframe storage and search in Azure AI Search using AzureSearchProvider."""

    def __init__(self, search_endpoint: str, index_name: str = "video-keyframes-index"):
        """
        Initialize the keyframe search index.

        Args:
            search_endpoint: Azure AI Search endpoint URL
            index_name: Name of the search index for keyframes
        """
        self.search_endpoint = search_endpoint
        self.index_name = index_name

        # Initialize Azure Search provider
        config = {
            "endpoint": search_endpoint,
            "index_name": index_name,
            "use_managed_identity": True
        }
        self.provider = AzureSearchProvider(config)

    async def create_keyframe_index_if_not_exists(self) -> bool:
        """
        Create the keyframe index with the proper schema if it doesn't exist.

        Returns:
            bool: True if index was created, False if it already existed
        """
        try:
            # Check if index exists
            if await self.provider.index_exists(self.index_name):
                logger.info(f"Keyframe index '{self.index_name}' already exists")
                return False

            # Define keyframe-specific fields (matching reference schema)
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="video_id", type=SearchFieldDataType.String,
                              filterable=True, facetable=True),
                SearchableField(name="keyframe_filename", type=SearchFieldDataType.String,
                              filterable=True, facetable=True),
                SearchField(name="clip_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                           searchable=True, vector_search_dimensions=512,
                           vector_search_profile_name="clip-profile"),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset,
                           filterable=True, sortable=True),
                SimpleField(name="motion_score", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="timestamp_seconds", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="blob_url", type=SearchFieldDataType.String),
                SimpleField(name="parent_id", type=SearchFieldDataType.String,
                           filterable=True),
                SimpleField(name="parent_duration", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True),
                SimpleField(name="video_duration", type=SearchFieldDataType.Double,
                           filterable=True, sortable=True)
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

            return await self.provider.create_index(self.index_name, index)

        except Exception as e:
            logger.error(f"Failed to create keyframe index: {e}")
            raise

    async def upload_frame_embeddings(self, frame_embeddings: List[FrameEmbedding],
                                    video_id: str, video_path: str,
                                    parent_id: Optional[str] = None,
                                    parent_duration: Optional[float] = None,
                                    video_duration: Optional[float] = None) -> bool:
        """
        Upload frame embeddings to the search index.

        Args:
            frame_embeddings: List of FrameEmbedding objects
            video_id: Unique video identifier
            video_path: Path to the video file
            parent_id: Parent video ID (original video before splitting)
            parent_duration: Duration of parent video in seconds
            video_duration: Duration of this video part in seconds

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
            documents = create_frame_documents_from_embeddings(
                frame_embeddings, video_id, video_path,
                parent_id, parent_duration, video_duration
            )

            # Upload in batches using provider
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                # Upload each document in the batch
                for doc in batch:
                    await self.provider.index_document(doc, self.index_name)

                logger.info(f"Uploaded batch {i // batch_size + 1} of {len(batch)} frame documents")

            logger.info(f"Successfully uploaded {len(documents)} frame embeddings to search index")
            return True

        except Exception as e:
            logger.error(f"Failed to upload frame embeddings: {e}")
            return False

    async def close(self):
        """Close the provider."""
        if self.provider:
            await self.provider.close()