"""Video frame search utilities; supports pluggable search providers (Azure or local FAISS)."""

import numpy as np
from typing import List, Dict, Any, Optional
from azure.search.documents.models import VectorizedQuery

from mmct.providers.factory import provider_factory
from mmct.providers.base import SearchProvider


class VideoFrameSearchClient:
    """Video frame search client with a pluggable search provider.

    By default the provider is created via ProviderFactory (reads MMCTConfig). You can pass
    an explicit provider instance or a provider_name (e.g. 'local_faiss').
    """

    def __init__(
        self,
        search_endpoint: Optional[str] = None,
        search_key: Optional[str] = None,
        index_name: str = "video-frames-index",
        provider: Optional[SearchProvider] = None,
        provider_name: Optional[str] = None,
        provider_config: Optional[dict] = None,
    ):
        """Initialize the video frame search client.

        Args:
            search_endpoint: optional endpoint to inject into provider config
            search_key: optional api key to inject into provider config
            index_name: name of the search index
            provider: optional pre-created SearchProvider instance
            provider_name: optional provider name to create via ProviderFactory
            provider_config: optional provider config dict to pass to factory-created provider
        """
        if provider is not None:
            self.provider = provider
        else:
            # create provider via factory; provider_name may be None to use config default
            self.provider = provider_factory.create_search_provider(provider_name)

        # allow caller to override config values
        if provider_config:
            # merge provided config into provider.config (shallow)
            try:
                self.provider.config.update(provider_config)
            except Exception:
                self.provider.config = provider_config

        if search_endpoint:
            self.provider.config["endpoint"] = search_endpoint
        if search_key:
            self.provider.config["api_key"] = search_key

        self.index_name = index_name
        # keep provider aware of which index it will operate on
        self.provider.config["index_name"] = index_name
        if hasattr(self.provider, "_initialize_client"):
            # some providers expose _initialize_client() factory for client attribute
            try:
                self.provider.client = self.provider._initialize_client()
            except Exception:
                # ignore; provider may initialize lazily
                pass

    async def search_similar_frames(
        self,
        query_vector: np.ndarray,
        query_text: str = "",
        top_k: int = 10,
        filters: Optional[str] = None,
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
                fields="embeddings"
            )

            # Perform search using provider.
            results = await self.provider.search(
                query=query_text if query_text else "*",
                index_name=self.index_name,
                search_text=query_text if query_text else "*",
                vector_queries=[vector_query],
                embedding=vector_query.vector if hasattr(vector_query, "vector") else (vector_query.vector if isinstance(vector_query, dict) and "vector" in vector_query else query_vector),
                filter=filters,
                top=top_k,
                select=["keyframe_filename", "video_id", "timestamp_seconds", "motion_score"],
                query_type="vector",
            )

            return results

        except Exception as e:
            return []

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
    Create a frame document following KeyframeDocument schema.

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
        Document dictionary following KeyframeDocument schema
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
        "embeddings": clip_embedding.tolist() if isinstance(clip_embedding, np.ndarray) else clip_embedding,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_score": float(motion_score),
        "timestamp_seconds": float(timestamp_seconds)
    }

    return document
