"""
CLIP embeddings generation for video keyframes.
"""

import os
import logging
from typing import List, Optional
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

from ..core import FrameMetadata, FrameEmbedding, EmbeddingConfig, get_media_folder

logger = logging.getLogger(__name__)


class CLIPEmbeddingsGenerator:
    """Generate CLIP embeddings for video frames."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
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

            logger.info(f"Initializing Embedding model {self.config.clip_model_name} on {self.device}")

            # Load model and processor
            if self.config.clip_model_name.startswith('openai'):
                self.model = CLIPModel.from_pretrained(self.config.clip_model_name)
                self.processor = CLIPProcessor.from_pretrained(self.config.clip_model_name, use_fast=False)
            elif self.config.clip_model_name.startswith('vidore'):
                self.model = ColQwen2_5.from_pretrained(self.config.clip_model_name, attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,)
                self.processor = ColQwen2_5_Processor.from_pretrained(self.config.clip_model_name, use_fast=False)
            else:
                raise ValueError(f"Unsupported model name: {self.config.clip_model_name}")

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("Embedding model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Embedding model: {e}")
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
        try:
            if isinstance(self.processor, CLIPProcessor):
                # CLIP path
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
            else:
                # ColQwen2_5 path — use processor.process_images
                processed = self.processor.process_images(images)  # likely returns a dict or a tensor wrapper
                processed = {k: v.to(self.device) for k, v in processed.items()}
                with torch.no_grad():
                    outputs = self.model(**processed)
                    # depending on implementation, outputs might be a tensor or a dict
                    if hasattr(outputs, "image_embeds"):
                        feats = outputs.image_embeds
                    elif isinstance(outputs, torch.Tensor):
                        feats = outputs
                    else:
                        # Could be a dict or multi-vector structure
                        # e.g. feats = outputs["image_embeds"]
                        feats = outputs.get("image_embeds", None)
                        if feats is None:
                            raise RuntimeError("Could not find feature outputs in ColQwen2_5 forward output")

            # normalize
            feats = feats / feats.norm(dim=1, keepdim=True)

            return feats.cpu().numpy()

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    async def process_frames(
        self,
        frame_metadata_list: List[FrameMetadata],
        video_id: str
    ) -> List[FrameEmbedding]:
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

                # Load images for this batch
                for frame_metadata in batch_metadata:
                    frame_filename = f"{video_id}_{frame_metadata.frame_number}.jpg"
                    frame_path = os.path.join(keyframes_dir, frame_filename)

                    if os.path.exists(frame_path):
                        image = self._load_and_preprocess_image(frame_path)
                        if image is not None:
                            batch_images.append(image)
                            batch_valid_metadata.append(frame_metadata)
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
                    for metadata, embedding in zip(batch_valid_metadata, batch_embeddings):
                        frame_embedding = FrameEmbedding(
                            frame_metadata=metadata,
                            clip_embedding=embedding,
                            blob_url=None,
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
