from abc import ABC, abstractmethod
from typing import List, Union
from PIL import Image
import numpy as np

class ImageEmbeddingProvider(ABC):
    """Abstract base class for image embedding providers."""

    @abstractmethod
    async def image_embedding(self, image: Union[str, Image.Image], **kwargs) -> List[float]:
        """
        Generate embedding for a single image.

        Args:
            image: Either a file path (str) or PIL Image object
            **kwargs: Additional provider-specific parameters

        Returns:
            Image embedding as a list of floats
        """
        pass

    @abstractmethod
    async def batch_image_embedding(self, images: List[Union[str, Image.Image]], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple images.

        Args:
            images: List of image paths or PIL Image objects
            **kwargs: Additional provider-specific parameters

        Returns:
            List of image embeddings
        """
        pass

    @abstractmethod
    def close(self):
        """Close the provider and cleanup resources."""
        pass
