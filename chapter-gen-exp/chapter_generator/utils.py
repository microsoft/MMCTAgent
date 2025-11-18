"""
Utility functions for chapter generation pipeline.

Common helper functions used across chapter generation, subject registry, and video summary modules.
"""

from typing import List
from loguru import logger
from mmct.providers.factory import provider_factory


async def create_embedding(text: str) -> List[float]:
    """
    Create embedding vector for the given text.

    Args:
        text: Input text to generate embedding for

    Returns:
        List of floats representing the embedding vector

    Raises:
        Exception: If embedding creation fails
    """
    embedding_provider = None
    try:
        embedding_provider = provider_factory.create_embedding_provider()
        embedding = await embedding_provider.embedding(text)
        logger.debug(f"Created embedding for text of length {len(text)}")
        return embedding
    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        raise Exception(f"Failed to create embedding: {e}")
    finally:
        # Close provider to clean up resources
        if embedding_provider and hasattr(embedding_provider, 'close'):
            try:
                await embedding_provider.close()
                logger.debug("Embedding provider closed successfully")
            except Exception as close_error:
                logger.warning(f"Error closing embedding provider: {close_error}")
