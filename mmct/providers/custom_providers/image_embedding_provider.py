from mmct.providers.base import ImageEmbeddingProvider
from mmct.config.settings import ImageEmbeddingConfig
from typing import Dict, Any, List, Union, Optional
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from loguru import logger
from mmct.utils.error_handler import handle_exceptions, convert_exceptions, ProviderException, ConfigurationException
import asyncio


class CustomImageEmbeddingProvider(ImageEmbeddingProvider):
    """CLIP-based image and text embedding provider implementation."""

    def __init__(self, config: Union[Dict[str, Any], ImageEmbeddingConfig]):
        """
        Initialize CLIP image embedding provider.

        Args:
            config: ImageEmbeddingConfig object or dict with following keys:
                - model_name: CLIP model name (default: "openai/clip-vit-base-patch32")
                - device: Device to use - "auto", "cpu", or "cuda" (default: "auto")
                - max_image_size: Maximum image dimension (default: 224)
                - batch_size: Batch size for processing (default: 8)
                
        Note:
            Embeddings are always L2 normalized for optimal CLIP performance.
        """
        # Convert ImageEmbeddingConfig to dict if needed
        if isinstance(config, ImageEmbeddingConfig):
            config = config.to_provider_config()

        self.config = config
        self.model_name = config.get("model_name", "openai/clip-vit-base-patch32")
        self.device = self._get_device()
        self.max_image_size = config.get("max_image_size", 224)
        self.batch_size = config.get("batch_size", 8)

        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self._initialize_model()

    def _get_device(self) -> str:
        """Determine the best device to use."""
        device_config = self.config.get("device", "auto")

        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_config

    def _initialize_model(self):
        """Initialize the CLIP model and processor."""
        try:
            logger.info(f"Initializing CLIP model {self.model_name} on {self.device}")

            # Load model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, use_fast=False)

            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info("CLIP model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise ConfigurationException(f"Failed to initialize CLIP model: {e}")

    def _load_and_preprocess_image(self, image: Union[str, Image.Image]) -> Optional[Image.Image]:
        """
        Load and preprocess an image.

        Args:
            image: Either a file path (str) or PIL Image object

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            else:
                img = image.convert('RGB') if image.mode != 'RGB' else image

            # Resize if too large
            if max(img.size) > self.max_image_size:
                img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)

            return img

        except Exception as e:
            logger.warning(f"Failed to load/preprocess image: {e}")
            return None

    def _generate_embeddings_sync(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate embeddings for a batch of images (synchronous).

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
                
                # Always L2 normalize embeddings for CLIP
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Convert to numpy and move to CPU
            embeddings = image_features.cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise ProviderException(f"Failed to generate batch embeddings: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def image_embedding(self, image: Union[str, Image.Image], **kwargs) -> List[float]:
        """
        Generate embedding for a single image.

        Args:
            image: Either a file path (str) or PIL Image object
            **kwargs: Additional parameters (unused for CLIP)

        Returns:
            Image embedding as a list of floats
        """
        try:
            # Load and preprocess image
            img = self._load_and_preprocess_image(image)

            if img is None:
                raise ProviderException("Failed to load or preprocess image")

            # Run embedding generation in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self._generate_embeddings_sync,
                [img]
            )

            return embeddings[0].tolist()

        except Exception as e:
            logger.error(f"CLIP image embedding failed: {e}")
            raise ProviderException(f"CLIP image embedding failed: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_image_embedding(self, images: List[Union[str, Image.Image]], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple images.

        Args:
            images: List of image paths or PIL Image objects
            **kwargs: Additional parameters (unused for CLIP)

        Returns:
            List of image embeddings
        """
        try:
            # Load and preprocess all images
            processed_images = []
            for image in images:
                img = self._load_and_preprocess_image(image)
                if img is not None:
                    processed_images.append(img)
                else:
                    logger.warning(f"Skipping failed image in batch")

            if not processed_images:
                raise ProviderException("No valid images to process in batch")

            # Run embedding generation in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self._generate_embeddings_sync,
                processed_images
            )

            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"CLIP batch image embedding failed: {e}")
            raise ProviderException(f"CLIP batch image embedding failed: {e}")

    def _generate_text_embeddings_sync(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts (synchronous).

        Args:
            texts: List of text strings

        Returns:
            NumPy array of embeddings
        """
        try:
            # Process texts
            inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
                # Always L2 normalize embeddings for CLIP
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Convert to numpy and move to CPU
            embeddings = text_features.cpu().numpy()

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate text embeddings: {e}")
            raise ProviderException(f"Failed to generate text embeddings: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def text_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string
            **kwargs: Additional parameters (unused for CLIP)

        Returns:
            Text embedding as a list of floats
        """
        try:
            # Run embedding generation in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self._generate_text_embeddings_sync,
                [text]
            )

            return embeddings[0].tolist()

        except Exception as e:
            logger.error(f"CLIP text embedding failed: {e}")
            raise ProviderException(f"CLIP text embedding failed: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_text_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of text strings
            **kwargs: Additional parameters (unused for CLIP)

        Returns:
            List of text embeddings
        """
        try:
            # Run embedding generation in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self._generate_text_embeddings_sync,
                texts
            )

            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"CLIP batch text embedding failed: {e}")
            raise ProviderException(f"CLIP batch text embedding failed: {e}")

    def close(self):
        """Close the provider and cleanup resources."""
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.processor is not None:
                del self.processor
                self.processor = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("CLIP embedding provider cleaned up successfully")

        except Exception as e:
            logger.warning(f"Error during CLIP provider cleanup: {e}")
