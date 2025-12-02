"""Common embedding utilities for CLIP-based text and image embeddings."""
import numpy as np
from mmct.providers.custom_providers import ClipImageEmbeddingProvider


class EmbeddingsGenerator:
    """Generate CLIP embeddings for text (shared utility)."""

    def __init__(self, image_embedding_provider: ClipImageEmbeddingProvider):
        """
        Initialize the embeddings generator.

        Args:
            image_embedding_provider: ClipImageEmbeddingProvider instance
        """
        self.provider = image_embedding_provider

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
