"""Local embedding provider using SentenceTransformer."""

import asyncio
from typing import List, Optional
from loguru import logger

from mmct.providers.base import BaseEmbeddingProvider


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using SentenceTransformer models.
    
    Provides text embeddings using pre-trained sentence transformer models
    running locally without external API calls.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """Initialize the local embedding provider.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
                Default is 'all-MiniLM-L6-v2' (384 dimensions, fast).
                Other options: 'all-mpnet-base-v2' (768 dims, higher quality).
            device: Device to run model on ('cpu', 'cuda', 'mps').
                None for auto-detection.
            normalize_embeddings: Whether to L2-normalize embeddings.
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._lock = asyncio.Lock()
    
    def _ensure_model_loaded(self) -> None:
        """Lazily load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Model loaded successfully. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
            except ImportError as e:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise ImportError(
                    "sentence-transformers package is required. "
                    "Install with: pip install sentence-transformers"
                ) from e
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding synchronously.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        self._ensure_model_loaded()
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def _generate_batch_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings synchronously.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        self._ensure_model_loaded()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            batch_size=32
        )
        return [emb.tolist() for emb in embeddings]
    
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate text embedding asynchronously.
        
        Args:
            text: Text to generate embedding for.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            Embedding vector as list of floats.
        """
        async with self._lock:
            return await asyncio.to_thread(self._generate_embedding_sync, text)
    
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts asynchronously.
        
        Args:
            texts: List of texts to generate embeddings for.
            **kwargs: Additional parameters (ignored).
            
        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        
        async with self._lock:
            return await asyncio.to_thread(self._generate_batch_embeddings_sync, texts)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the loaded model.
        
        Returns:
            Embedding vector dimension.
        """
        self._ensure_model_loaded()
        return self._model.get_sentence_embedding_dimension()
    
    async def close(self) -> None:
        """Cleanup model resources."""
        self._model = None
        logger.info("LocalEmbeddingProvider closed")
