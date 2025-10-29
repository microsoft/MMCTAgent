"""Video frame search utilities using Azure Search provider."""
import numpy as np
from typing import List, Dict, Any, Optional
from azure.search.documents.models import VectorizedQuery

from mmct.providers.azure_providers.search_provider import AzureSearchProvider


class VideoFrameSearchClient:
    """Video frame search client using AzureSearchProvider."""

    def __init__(self, search_endpoint: str, search_key: Optional[str] = None,
                 index_name: str = "video-frames-index"):
        """
        Initialize the video frame search client.

        Args:
            search_endpoint: Azure AI Search endpoint URL
            search_key: API key (if None, uses managed identity)
            index_name: Name of the search index
        """
        config = {
            "endpoint": search_endpoint,
            "index_name": index_name,
            "use_managed_identity": search_key is None,
        }
        if search_key:
            config["api_key"] = search_key

        self.provider = AzureSearchProvider(config)
        self.index_name = index_name

    async def search_similar_frames(
        self,
        query_vector: np.ndarray,
        query_text: str = "",
        top_k: int = 10,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar frames using vector search.

        Args:
            query_vector: CLIP embedding vector for similarity search
            query_text: Optional text query for hybrid search
            top_k: Number of results to return
            filters: Optional OData filter expression

        Returns:
            List of search results
        """
        try:
            # Create vector query for CLIP embeddings
            vector_query = VectorizedQuery(
                vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
                k_nearest_neighbors=top_k,
                fields="clip_embedding"
            )

            # Perform search using provider
            results = await self.provider.search(
                query=query_text if query_text else "*",
                index_name=self.index_name,
                search_text=query_text if query_text else "*",
                vector_queries=[vector_query],
                filter=filters,
                top=top_k,
                select=['keyframe_filename', 'video_id', 'timestamp_seconds', 'motion_score'],
                query_type="vector"
            )

            return results

        except Exception as e:
            return []

    async def upload_frame_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upload frame documents to the search index.

        Args:
            documents: List of frame documents to upload

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                return True

            # Upload documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                for doc in batch:
                    await self.provider.index_document(doc, self.index_name)

            return True

        except Exception as e:
            return False

    async def delete_frames_by_video(self, video_id: str) -> bool:
        """
        Delete all frames for a specific video.

        Args:
            video_id: Video ID to delete frames for

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First, get all frame IDs for this video
            filter_expr = f"video_id eq '{video_id}'"
            frames = await self.provider.search(
                query="*",
                index_name=self.index_name,
                search_text="*",
                filter=filter_expr,
                top=1000,
                select=['id']
            )

            if not frames:
                return True

            # Delete each document
            for frame in frames:
                await self.provider.delete_document(frame["id"], self.index_name)

            return True

        except Exception as e:
            return False

    async def close(self):
        """Close the provider and cleanup resources."""
        if self.provider:
            await self.provider.close()


def create_frame_documents_from_embeddings(
    frame_embeddings: List,
    video_id: str,
    video_path: str,
    parent_id: Optional[str] = None,
    parent_duration: Optional[float] = None,
    video_duration: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Create search documents from frame embeddings.

    Args:
        frame_embeddings: List of FrameEmbedding objects
        video_id: Unique video identifier
        video_path: Path to the video file
        parent_id: Parent video ID (original video before splitting)
        parent_duration: Duration of parent video in seconds
        video_duration: Duration of this video part in seconds

    Returns:
        List of document dictionaries ready for Azure AI Search
    """
    import uuid
    from datetime import datetime, timezone

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
            "clip_embedding": frame_embedding.clip_embedding.tolist() if isinstance(frame_embedding.clip_embedding, np.ndarray) else frame_embedding.clip_embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "motion_score": float(frame_embedding.frame_metadata.motion_score),
            "timestamp_seconds": float(frame_embedding.frame_metadata.timestamp_seconds),
            "blob_url": "",  # Can be populated if needed
            "parent_id": parent_id if parent_id else video_id,
            "parent_duration": parent_duration if parent_duration is not None else 0.0,
            "video_duration": video_duration if video_duration is not None else 0.0
        }

        documents.append(document)

    return documents


def create_frame_document(
    video_path: str,
    frame_path: str,
    frame_number: int,
    timestamp_seconds: float,
    motion_score: float,
    width: int,
    height: int,
    clip_embedding: np.ndarray,
    extracted_text: str = "",
    tags: List[str] = None,
    video_id: str = None
) -> Dict[str, Any]:
    """
    Create a frame document for Azure AI Search.

    Args:
        video_path: Path to the source video file
        frame_path: Path to the extracted frame image
        frame_number: Frame number in the video
        timestamp_seconds: Time position in seconds
        motion_score: Optical flow motion score
        width: Frame width in pixels
        height: Frame height in pixels
        clip_embedding: CLIP embedding vector
        extracted_text: Any OCR text found in frame
        tags: Optional tags for the frame
        video_id: Hash-based video ID (if None, uses filename)

    Returns:
        Document dictionary ready for Azure AI Search
    """
    import os
    import uuid
    from datetime import datetime, timezone

    # Generate unique ID
    frame_id = f"{os.path.basename(video_path)}_{frame_number}_{int(timestamp_seconds)}"
    frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, frame_id))

    # Use provided video_id (hash) or fallback to filename
    if video_id is None:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Generate keyframe filename using video_id
    keyframe_filename = f"{video_id}_{frame_number}.jpg"

    document = {
        "id": frame_id,
        "video_id": video_id,
        "keyframe_filename": keyframe_filename,
        "clip_embedding": clip_embedding.tolist() if isinstance(clip_embedding, np.ndarray) else clip_embedding,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_score": float(motion_score),
        "timestamp_seconds": float(timestamp_seconds)
    }

    return document
