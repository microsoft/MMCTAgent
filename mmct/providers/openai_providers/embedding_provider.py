
from mmct.providers.base import EmbeddingProvider
from typing import Dict, Any, List
from loguru import logger
from mmct.utils.error_handler import handle_exceptions, convert_exceptions, ProviderException, ConfigurationException
from openai import AsyncOpenAI, OpenAI



class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                raise ConfigurationException("OpenAI API key is required")
            
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            return AsyncOpenAI(
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using OpenAI."""
        try:
            model = self.config.get("embedding_model", "text-embedding-3-small")
            
            response = await self.client.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise ProviderException(f"OpenAI embedding failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        try:
            model = self.config.get("embedding_model", "text-embedding-3-small")
            
            response = await self.client.embeddings.create(
                model=model,
                input=texts,
                **kwargs
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            raise ProviderException(f"OpenAI batch embedding failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct embeddings API access."""
        return self.client

    async def close(self):
        """Close the embedding client and cleanup resources."""
        if self.client:
            logger.info("Closing OpenAI embedding client")
            await self.client.close()
