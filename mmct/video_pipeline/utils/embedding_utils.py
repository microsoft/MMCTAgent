"""Common embedding utilities for CLIP-based text and image embeddings."""
import numpy as np
from typing import Optional

from mmct.providers.custom_providers import CustomImageEmbeddingProvider
from mmct.config.settings import ImageEmbeddingConfig


class EmbeddingsGenerator:
    """Generate CLIP embeddings for text (shared utility)."""

    def __init__(self, config: Optional[ImageEmbeddingConfig] = None):
        """
        Initialize the embeddings generator.

        Args:
            config: ImageEmbeddingConfig object for embedding parameters
        """
        self.config = config or ImageEmbeddingConfig()

        # Initialize the embedding provider
        self.provider = CustomImageEmbeddingProvider(self.config)

    async def generate_text_embedding(self, text: str) -> np.ndarray:
        """
        Generate CLIP embedding for text.

        Args:
            text: Input text string

        Returns:
            CLIP text embedding as numpy array
        """
        try:
            # Use the provider to generate text embedding
            embedding = await self.provider.text_embedding(text)
            return np.array(embedding)

        except Exception as e:
            return np.zeros(512, dtype=np.float32)

    async def close(self):
        """Close the provider and cleanup resources."""
        if self.provider:
            await self.provider.close()
