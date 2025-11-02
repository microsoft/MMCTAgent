from typing import List, Optional
import logging

from mmct.providers.factory import provider_factory
from mmct.video_pipeline.core.ingestion.key_frames_extractor.clip_embeddings import FrameEmbedding
from mmct.video_pipeline.utils.video_frame_search import create_frame_documents_from_embeddings
from mmct.providers.search_index_schema import create_keyframe_index_schema

logger = logging.getLogger(__name__)


class KeyframeSearchIndex:
    """Provider-agnostic keyframe storage and indexing helper."""

    def __init__(self, search_endpoint: str = None, index_name: str = "video-keyframes-index", provider_name: Optional[str] = None, provider_config: Optional[dict] = None):
        self.search_endpoint = search_endpoint
        self.index_name = index_name

        # create provider via factory (uses configured default if provider_name is None)
        self.provider = provider_factory.create_search_provider(provider_name)

        # merge provided config and endpoint
        cfg = provider_config or {}
        if search_endpoint:
            cfg.setdefault("endpoint", search_endpoint)
        cfg.setdefault("index_name", index_name)
        try:
            self.provider.config.update(cfg)
        except Exception:
            self.provider.config = cfg

        # initialize client shim when available
        if hasattr(self.provider, "_initialize_client"):
            try:
                self.provider.client = self.provider._initialize_client()
            except Exception:
                pass

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

            # If provider is Azure, create the rich SearchIndex schema via the helper
            if provider_factory.is_search_provider(self.provider, "azure_ai_search"):
                index_schema = create_keyframe_index_schema(self.index_name, dim=512)
                return await self.provider.create_index(self.index_name, index_schema)

            # Non-Azure (e.g., local FAISS) — pass a simple schema dict and let provider decide
            return await self.provider.create_index(self.index_name, {"dim": None})

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
                frame_embeddings, video_id, video_path, parent_id, parent_duration, video_duration
            )

            # Detect provider type and transform for non-Azure providers
            is_azure = provider_factory.is_search_provider(self.provider, "azure_ai_search")

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                if not is_azure:
                    # map clip_embedding -> embeddings for FAISS-style providers
                    transformed = []
                    for doc in batch:
                        new_doc = dict(doc)
                        if "clip_embedding" in new_doc:
                            new_doc["embeddings"] = new_doc.get("clip_embedding")
                        transformed.append(new_doc)

                    if hasattr(self.provider, "upload_documents"):
                        await self.provider.upload_documents(transformed, index_name=self.index_name)
                    else:
                        for doc in transformed:
                            await self.provider.index_document(doc, self.index_name)

                else:
                    # Azure expects clip_embedding field; prefer bulk API
                    if hasattr(self.provider, "upload_documents"):
                        await self.provider.upload_documents(batch, index_name=self.index_name)
                    else:
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