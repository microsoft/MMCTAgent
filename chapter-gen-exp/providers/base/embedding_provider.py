from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate text embedding."""
        pass
    
    @abstractmethod
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass