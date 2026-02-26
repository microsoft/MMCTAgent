"""
FastEmbed QdrantCLIP Image Embedding Provider

Uses Qdrant's CLIP ViT-B-32 model via FastEmbed for CPU-optimized image embeddings.
Supports both image and text embeddings in the same embedding space for cross-modal retrieval.

Model: Qdrant/clip-ViT-B-32-vision (512 dimensions)
- ONNX-optimized for fast CPU inference (~170 images/sec)
- L2-normalized embeddings for cosine similarity
- Shared embedding space with CLIP text encoder

Usage:
    provider = FastEmbedQdrantCLIPEmbeddingProvider()
    
    # Image embedding
    embedding = await provider.image_embedding("keyframe.jpg")
    
    # Text embedding (for text-to-image search)
    text_emb = await provider.text_embedding("a person walking")
    
    # Batch processing
    embeddings = await provider.batch_image_embedding(["img1.jpg", "img2.jpg"])
"""

from mmct.providers.base import BaseImageEmbeddingProvider
from typing import List, Union, Optional
from PIL import Image
import numpy as np
from loguru import logger
from mmct.utils.error_handler import (
    handle_exceptions,
    convert_exceptions,
    ProviderException,
    ConfigurationException,
)
import asyncio

# Lazy imports to avoid loading at module level
_image_embedding_model = None
_text_embedding_model = None


def _get_image_embedding_model():
    """Lazy load the FastEmbed image embedding model (singleton)."""
    global _image_embedding_model
    if _image_embedding_model is None:
        try:
            from fastembed import ImageEmbedding

            logger.info("Loading FastEmbed QdrantCLIP image model: Qdrant/clip-ViT-B-32-vision")
            _image_embedding_model = ImageEmbedding(
                model_name="Qdrant/clip-ViT-B-32-vision",
                providers=["CPUExecutionProvider"],
            )
            logger.info("FastEmbed QdrantCLIP image model loaded (512-dim, ONNX/CPU)")
        except Exception as e:
            logger.error(f"Failed to load FastEmbed image model: {e}")
            raise ConfigurationException(f"Failed to load FastEmbed image model: {e}")
    return _image_embedding_model


def _get_text_embedding_model():
    """Lazy load the FastEmbed text embedding model for CLIP (singleton)."""
    global _text_embedding_model
    if _text_embedding_model is None:
        try:
            from fastembed import TextEmbedding

            # Use CLIP text encoder that shares embedding space with the image model
            logger.info("Loading FastEmbed QdrantCLIP text model: Qdrant/clip-ViT-B-32-text")
            _text_embedding_model = TextEmbedding(
                model_name="Qdrant/clip-ViT-B-32-text",
                providers=["CPUExecutionProvider"],
            )
            logger.info("FastEmbed QdrantCLIP text model loaded (512-dim, ONNX/CPU)")
        except Exception as e:
            logger.error(f"Failed to load FastEmbed text model: {e}")
            raise ConfigurationException(f"Failed to load FastEmbed text model: {e}")
    return _text_embedding_model


class FastEmbedQdrantCLIPEmbeddingProvider(BaseImageEmbeddingProvider):
    """
    FastEmbed-based CLIP image and text embedding provider.
    
    Uses Qdrant/clip-ViT-B-32 models via ONNX Runtime for CPU-optimized inference.
    Produces 512-dimensional L2-normalized embeddings suitable for cosine similarity search.
    
    Key features:
    - CPU-optimized via ONNX Runtime (~170 images/sec throughput)
    - Shared embedding space for text and images (cross-modal retrieval)
    - Lazy model loading (models loaded on first use)
    - Singleton pattern for memory efficiency
    
    Attributes:
        EMBEDDING_DIM: Output embedding dimension (512)
        IMAGE_MODEL: Qdrant/clip-ViT-B-32-vision
        TEXT_MODEL: Qdrant/clip-ViT-B-32-text
    """

    EMBEDDING_DIM = 512
    IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"
    TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"

    def __init__(self, batch_size: int = 32, max_image_size: Optional[int] = None):
        """
        Initialize the FastEmbed QdrantCLIP provider.
        
        Args:
            batch_size: Batch size for processing multiple images (default: 32)
            max_image_size: Optional max image dimension for resizing (default: None, uses model default)
        
        Note:
            Models are loaded lazily on first use to reduce startup time.
            Uses CPUExecutionProvider for ONNX Runtime.
        """
        self.batch_size = batch_size
        self.max_image_size = max_image_size
        self._initialized = False

    def _ensure_initialized(self):
        """Ensure models are loaded (lazy initialization)."""
        if not self._initialized:
            # Trigger lazy loading
            _get_image_embedding_model()
            self._initialized = True

    def _load_and_preprocess_image(self, image: Union[str, Image.Image]) -> Optional[str]:
        """
        Load and preprocess an image, returning path for FastEmbed.
        
        FastEmbed's ImageEmbedding accepts file paths directly.
        For PIL images, we need to save to a temp file or pass the PIL object.
        
        Args:
            image: Either a file path (str) or PIL Image object
            
        Returns:
            File path string or None if processing fails
        """
        try:
            if isinstance(image, str):
                # FastEmbed accepts file paths directly
                return image
            elif isinstance(image, Image.Image):
                # FastEmbed also accepts PIL images
                return image
            else:
                logger.warning(f"Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            logger.warning(f"Failed to preprocess image: {e}")
            return None

    def _generate_image_embeddings_sync(
        self, images: List[Union[str, Image.Image]]
    ) -> np.ndarray:
        """
        Generate embeddings for images synchronously.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            NumPy array of embeddings (N x 512)
        """
        try:
            model = _get_image_embedding_model()
            
            # FastEmbed's embed() returns a generator, convert to list
            embeddings = list(model.embed(images))
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to generate image embeddings: {e}")
            raise ProviderException(f"Failed to generate image embeddings: {e}")

    def _generate_text_embeddings_sync(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts synchronously.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (N x 512)
        """
        try:
            model = _get_text_embedding_model()
            
            # FastEmbed's embed() returns a generator, convert to list
            embeddings = list(model.embed(texts))
            
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to generate text embeddings: {e}")
            raise ProviderException(f"Failed to generate text embeddings: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def image_embedding(
        self, image: Union[str, Image.Image], **kwargs
    ) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: Either a file path (str) or PIL Image object
            **kwargs: Additional parameters (unused)
            
        Returns:
            Image embedding as a list of 512 floats (L2-normalized)
        """
        try:
            processed = self._load_and_preprocess_image(image)
            if processed is None:
                raise ProviderException("Failed to load or preprocess image")

            # Run in thread pool to avoid blocking
            embeddings = await asyncio.to_thread(
                self._generate_image_embeddings_sync, [processed]
            )

            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"QdrantCLIP image embedding failed: {e}")
            raise ProviderException(f"QdrantCLIP image embedding failed: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_image_embedding(
        self, images: List[Union[str, Image.Image]], **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of image paths or PIL Image objects
            **kwargs: Additional parameters (unused)
            
        Returns:
            List of image embeddings (each 512 floats, L2-normalized)
        """
        try:
            # Preprocess all images
            processed_images = []
            for image in images:
                processed = self._load_and_preprocess_image(image)
                if processed is not None:
                    processed_images.append(processed)
                else:
                    logger.warning("Skipping failed image in batch")

            if not processed_images:
                raise ProviderException("No valid images to process in batch")

            # Process in batches
            all_embeddings = []
            for i in range(0, len(processed_images), self.batch_size):
                batch = processed_images[i : i + self.batch_size]
                embeddings = await asyncio.to_thread(
                    self._generate_image_embeddings_sync, batch
                )
                all_embeddings.extend(embeddings)

            return [emb.tolist() for emb in all_embeddings]
        except Exception as e:
            logger.error(f"QdrantCLIP batch image embedding failed: {e}")
            raise ProviderException(f"QdrantCLIP batch image embedding failed: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def text_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding for a single text.
        
        Useful for text-to-image search in the shared CLIP embedding space.
        
        Args:
            text: Input text string
            **kwargs: Additional parameters (unused)
            
        Returns:
            Text embedding as a list of 512 floats (L2-normalized)
        """
        try:
            embeddings = await asyncio.to_thread(
                self._generate_text_embeddings_sync, [text]
            )
            return embeddings[0].tolist()
        except Exception as e:
            logger.error(f"QdrantCLIP text embedding failed: {e}")
            raise ProviderException(f"QdrantCLIP text embedding failed: {e}")

    @handle_exceptions(retries=2, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_text_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings
            **kwargs: Additional parameters (unused)
            
        Returns:
            List of text embeddings (each 512 floats, L2-normalized)
        """
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                embeddings = await asyncio.to_thread(
                    self._generate_text_embeddings_sync, batch
                )
                all_embeddings.extend(embeddings)

            return [emb.tolist() for emb in all_embeddings]
        except Exception as e:
            logger.error(f"QdrantCLIP batch text embedding failed: {e}")
            raise ProviderException(f"QdrantCLIP batch text embedding failed: {e}")

    def close(self):
        """
        Close the provider and cleanup resources.
        
        Note: Models are singletons shared across instances, so we don't
        delete them here. They will be garbage collected when the process ends.
        """
        logger.info("FastEmbed QdrantCLIP provider closed")


# Singleton accessor for convenient access
_provider_instance: Optional[FastEmbedQdrantCLIPEmbeddingProvider] = None


def get_qdrant_clip_provider() -> FastEmbedQdrantCLIPEmbeddingProvider:
    """
    Get or create the singleton FastEmbedQdrantCLIPEmbeddingProvider instance.
    
    Returns:
        FastEmbedQdrantCLIPEmbeddingProvider instance
    """
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = FastEmbedQdrantCLIPEmbeddingProvider()
    return _provider_instance
