"""FastEmbed Arctic embedding provider using Snowflake Arctic.

Provides 384-dimensional embeddings optimized for retrieval.
"""

from typing import List

from loguru import logger

from mmct.providers.base import BaseEmbeddingProvider

# Model outputs 384-dimensional embeddings (fixed by model architecture)
MODEL_NAME = "snowflake/snowflake-arctic-embed-s"


class FastEmbedArcticEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using Snowflake Arctic via fastembed.
    
    Uses snowflake/snowflake-arctic-embed-s for generating 384-dimensional
    embeddings optimized for retrieval.
    
    Key characteristics:
    - Model: snowflake/snowflake-arctic-embed-s
    - Dimensions: 384 (fixed by model architecture)
    - Execution: CPU via ONNXRuntime CPUExecutionProvider
    """
    
    def __init__(self):
        """Initialize the object embedding provider."""
        self._model = None
        self._model_name = MODEL_NAME
    
    def _load_model(self):
        """Lazy load the fastembed model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                logger.info(f"Loading object embedding model: {self._model_name}")
                self._model = TextEmbedding(
                    model_name=self._model_name,
                    providers=["CPUExecutionProvider"]
                )
                logger.info("Object embedding model loaded (dim=384)")
            except ImportError:
                raise ImportError(
                    "fastembed is required for local embeddings. "
                    "Install with: pip install fastembed"
                )
    
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed (object name + attributes)
            
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
            logger.info("Object embedding model unloaded")
