import os
import logging
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import FrameMetadata
from mmct.video_pipeline.utils.helper import get_media_folder

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    clip_model_name: str = "openai/clip-vit-base-patch32"
    batch_size: int = 8
    device: str = "auto"  # "cpu", "cuda", or "auto"
    max_image_size: int = 224

@dataclass
class FrameEmbedding:
    """Container for frame metadata and its embedding."""
    frame_metadata: FrameMetadata
    clip_embedding: np.ndarray
    frame_path: str
    tags: Optional[List[str]] = None

class CLIPEmbeddingsGenerator:
    """Generate CLIP embeddings for video frames."""

    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the embeddings generator.

        Args:
            config: Configuration object for embedding parameters
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.processor = None
        self.device = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the CLIP model and processor."""
        try:
            # Determine device
            if self.config.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self.config.device

            logger.info(f"Initializing CLIP model {self.config.clip_model_name} on {self.device}")

            # Load model and processor
            self.model = CLIPModel.from_pretrained(self.config.clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name, use_fast=False)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("CLIP model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise

    def _load_and_preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load and preprocess an image.

        Args:
            image_path: Path to the image file

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            image = Image.open(image_path).convert('RGB')

            # Resize if too large
            if max(image.size) > self.config.max_image_size:
                image.thumbnail((self.config.max_image_size, self.config.max_image_size), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _generate_batch_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.

        Args:
            images: List of PIL Image objects

        Returns:
            NumPy array of embeddings
        """
        try:
            # Process images
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Convert to numpy and move to CPU
            embeddings = image_features.cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

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
                batch_images = []
                batch_valid_metadata = []
                batch_frame_paths = []

                # Load images for this batch
                for frame_metadata in batch_metadata:
                    frame_filename = f"{video_id}_{frame_metadata.frame_number}.jpg"
                    frame_path = os.path.join(keyframes_dir, frame_filename)

                    if os.path.exists(frame_path):
                        image = self._load_and_preprocess_image(frame_path)
                        if image is not None:
                            batch_images.append(image)
                            batch_valid_metadata.append(frame_metadata)
                            batch_frame_paths.append(frame_path)
                        else:
                            logger.warning(f"Failed to load frame: {frame_path}")
                    else:
                        logger.warning(f"Frame file not found: {frame_path}")

                if not batch_images:
                    logger.warning(f"No valid images in batch {i // self.config.batch_size + 1}")
                    continue

                # Generate embeddings for this batch
                try:
                    batch_embeddings = self._generate_batch_embeddings(batch_images)

                    # Create FrameEmbedding objects
                    for metadata, embedding, frame_path in zip(batch_valid_metadata, batch_embeddings, batch_frame_paths):
                        frame_embedding = FrameEmbedding(
                            frame_metadata=metadata,
                            clip_embedding=embedding,
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

    def cleanup(self):
        """Clean up model resources."""
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Embeddings generator cleaned up successfully")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")