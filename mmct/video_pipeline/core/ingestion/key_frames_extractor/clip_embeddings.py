import os
import logging
from typing import List, Optional
import numpy as np
from dataclasses import dataclass
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import FrameMetadata
from mmct.video_pipeline.utils.helper import get_media_folder
from mmct.providers.custom_providers import CustomImageEmbeddingProvider
from mmct.config.settings import ImageEmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class FrameEmbedding:
    """Container for frame metadata and its embedding."""
    frame_metadata: FrameMetadata
    clip_embedding: np.ndarray
    frame_path: str
    tags: Optional[List[str]] = None

class CLIPEmbeddingsGenerator:
    """Generate CLIP embeddings for video frames."""

    def __init__(self, config: ImageEmbeddingConfig = None):
        """
        Initialize the embeddings generator.

        Args:
            config: ImageEmbeddingConfig object for embedding parameters
        """
        self.config = config or ImageEmbeddingConfig()

        # Initialize the image embedding provider
        self.provider = CustomImageEmbeddingProvider(self.config)

    async def process_frames(self, frame_metadata_list: List[FrameMetadata],
                           video_id: str) -> List[FrameEmbedding]:
        """
        Process a list of frame metadata and generate embeddings.

        Args:
            frame_metadata_list: List of FrameMetadata objects
            video_id: Video ID for finding frame files

        Returns:
            List of FrameEmbedding objects
        """
        if not frame_metadata_list:
            return []

        try:
            logger.info(f"Processing {len(frame_metadata_list)} frames for embedding generation")

            # Get keyframes directory
            media_folder = await get_media_folder()
            keyframes_dir = os.path.join(media_folder, "keyframes", video_id)

            frame_embeddings = []

            # Process frames in batches
            for i in range(0, len(frame_metadata_list), self.config.batch_size):
                batch_metadata = frame_metadata_list[i:i + self.config.batch_size]
                batch_frame_paths = []
                batch_valid_metadata = []

                # Collect valid frame paths for this batch
                for frame_metadata in batch_metadata:
                    frame_filename = f"{video_id}_{frame_metadata.frame_number}.jpg"
                    frame_path = os.path.join(keyframes_dir, frame_filename)

                    if os.path.exists(frame_path):
                        batch_frame_paths.append(frame_path)
                        batch_valid_metadata.append(frame_metadata)
                    else:
                        logger.warning(f"Frame file not found: {frame_path}")

                if not batch_frame_paths:
                    logger.warning(f"No valid images in batch {i // self.config.batch_size + 1}")
                    continue

                # Generate embeddings for this batch using the provider
                try:
                    batch_embeddings = await self.provider.batch_image_embedding(batch_frame_paths)

                    # Create FrameEmbedding objects
                    for metadata, embedding, frame_path in zip(batch_valid_metadata, batch_embeddings, batch_frame_paths):
                        frame_embedding = FrameEmbedding(
                            frame_metadata=metadata,
                            clip_embedding=np.array(embedding),
                            frame_path=frame_path,
                            tags=None  # Can be extended to include auto-generated tags
                        )
                        frame_embeddings.append(frame_embedding)

                    logger.info(f"Generated embeddings for batch {i // self.config.batch_size + 1} "
                              f"({len(batch_embeddings)} embeddings)")

                except Exception as e:
                    logger.error(f"Failed to process batch {i // self.config.batch_size + 1}: {e}")
                    continue

            logger.info(f"Successfully generated {len(frame_embeddings)} frame embeddings")
            return frame_embeddings

        except Exception as e:
            logger.error(f"Failed to process frames for embeddings: {e}")
            raise

    async def cleanup(self):
        """Clean up model resources."""
        try:
            if self.provider is not None:
                self.provider.close()

            logger.info("Embeddings generator cleaned up successfully")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")