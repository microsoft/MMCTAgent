"""FastEmbed BGE-Small embedding provider using BAAI/bge-small-en-v1.5.

Provides 384-dimensional embeddings optimized for semantic similarity.
"""

from typing import List

from loguru import logger

from mmct.providers.base import BaseEmbeddingProvider

# Model outputs 384-dimensional embeddings (fixed by model architecture)
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class FastEmbedBGEsmallEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using BGE-Small via fastembed.
    
    Uses BAAI/bge-small-en-v1.5 for generating 384-dimensional embeddings
    optimized for semantic similarity.
    
    Key characteristics:
    - Model: BAAI/bge-small-en-v1.5
    - Dimensions: 384 (fixed by model architecture)
    - Execution: CPU via ONNXRuntime CPUExecutionProvider
    """
    
    def __init__(self):
        """Initialize the event embedding provider."""
        self._model = None
        self._model_name = MODEL_NAME
    
    def _load_model(self):
        """Lazy load the fastembed model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                logger.info(f"Loading event embedding model: {self._model_name}")
                self._model = TextEmbedding(
                    model_name=self._model_name,
                    providers=["CPUExecutionProvider"]
                )
                logger.info("Event embedding model loaded (dim=384)")
            except ImportError:
                raise ImportError(
                    "fastembed is required for local embeddings. "
                    "Install with: pip install fastembed"
                )
    
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector (384-dim)
        """
        self._load_model()
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()
    
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors (384-dim each)
        """
        self._load_model()
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    
    def close(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Event embedding model unloaded")
