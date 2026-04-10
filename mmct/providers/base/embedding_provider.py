"""Base interface for text embedding providers.

This module defines the abstract interface for generating vector embeddings 
from text, which is used for semantic search within the MMCT graph database.
"""

from abc import ABC, abstractmethod
from typing import List, Any

class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding provider implementations (e.g., Azure OpenAI, Clip) must 
    inherit from this class to ensure interoperability across search components.
    """
    
    @abstractmethod
    async def embedding(self, text: str, **kwargs: Any) -> List[float]:
        """Generates a numerical embedding for a single text string.

        Args:
            text: The input text to embed.
            **kwargs: Additional provider-specific parameters.

        Returns:
            List[float]: The normalized vector embedding.
        """
        pass
    
    @abstractmethod
    async def batch_embedding(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generates numerical embeddings for multiple text strings.

        Args:
            texts: A list of input strings to embed.
            **kwargs: Additional provider-specific parameters.

        Returns:
            List[List[float]]: A list of normalized vector embeddings, one per 
                input string.
        """
        pass