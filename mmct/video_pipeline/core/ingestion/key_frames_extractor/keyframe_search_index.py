from typing import List, Optional, Dict, Any
import logging
import uuid
from datetime import datetime, timezone
import numpy as np

from mmct.providers.factory import provider_factory
from mmct.video_pipeline.core.ingestion.key_frames_extractor.clip_embeddings import FrameEmbedding

logger = logging.getLogger(__name__)


def create_frame_documents_from_embeddings(
    frame_embeddings: List[FrameEmbedding],
    video_id: str,
    video_path: str,
    parent_id: Optional[str] = None,
    parent_duration: Optional[float] = None,
    video_duration: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Create search documents from frame embeddings using KeyframeDocument model.

    Args:
        frame_embeddings: List of FrameEmbedding objects
        video_id: Unique video identifier
        video_path: Path to the video file
        parent_id: Parent video ID (original video before splitting)
        parent_duration: Duration of parent video in seconds
        video_duration: Duration of this video part in seconds

    Returns:
        List of document dictionaries following KeyframeDocument schema
    """
    documents = []

    for frame_embedding in frame_embeddings:
        # Generate unique ID for this frame
        frame_id = f"{video_id}_{frame_embedding.frame_metadata.frame_number}"
        frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, frame_id))

        # Create frame filename
        frame_filename = f"{video_id}_{frame_embedding.frame_metadata.frame_number}.jpg"

        # Create document following KeyframeDocument schema
        document = {
            "id": frame_id,
            "video_id": video_id,
            "keyframe_filename": frame_filename,
            "embeddings": frame_embedding.clip_embedding.tolist() if isinstance(frame_embedding.clip_embedding, np.ndarray) else frame_embedding.clip_embedding,
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


class KeyframeSearchIndex:
    """Provider-agnostic keyframe storage and indexing helper."""

    def __init__(self, search_endpoint: str = None, index_name: str = "video-keyframes-index", 
                 provider_name: Optional[str] = None, provider_config: Optional[dict] = None):
        """
        Initialize KeyframeSearchIndex with a search provider.
        
        Args:
            search_endpoint: Optional search endpoint (deprecated, use provider_config)
            index_name: Name of the index to use
            provider_name: Optional provider name (uses configured default if None)
            provider_config: Optional provider configuration dict
        """
        self.search_endpoint = search_endpoint
        self.index_name = index_name

        # Create provider via factory (uses configured default if provider_name is None)
        self.provider = provider_factory.create_search_provider(provider_name)

        # Apply additional configuration if provided
        if provider_config or search_endpoint:
            cfg = provider_config or {}
            if search_endpoint:
                cfg.setdefault("endpoint", search_endpoint)
            cfg.setdefault("index_name", index_name)
            
            # Merge config into provider config
            if hasattr(self.provider, 'config') and isinstance(self.provider.config, dict):
                self.provider.config.update(cfg)

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

            # Provider will handle schema creation based on type indicator
            return await self.provider.create_index(self.index_name, "keyframe")

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

            # Ensure index exists
            try:
                await self.create_keyframe_index_if_not_exists()
            except Exception as index_error:
                logger.warning(f"Index creation check failed: {index_error}")
                # Continue with upload anyway - index might already exist

            # Create documents using KeyframeDocument model
            documents = create_frame_documents_from_embeddings(
                frame_embeddings, video_id, video_path, parent_id, parent_duration, video_duration
            )

            # Upload in batches - provider handles field transformation
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Provider's upload_documents method will handle transformation
                await self.provider.upload_documents(batch, index_name=self.index_name)
                
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