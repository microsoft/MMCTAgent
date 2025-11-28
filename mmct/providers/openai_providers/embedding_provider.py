
from mmct.providers.base import BaseEmbeddingProvider
from typing import Dict, Any, List, Optional
from loguru import logger
from mmct.utils.error_handler import handle_exceptions, convert_exceptions, ProviderException, ConfigurationException
from openai import AsyncOpenAI, OpenAI



class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider implementation."""
    
    def __init__(self, api_key:str, model_name:str, timeout:Optional[int] = 200, max_retries:Optional[int] = 2):
        
        if not api_key:
                raise ConfigurationException("OpenAI API key is required!")
        
        if not model_name:
            raise ConfigurationException("OpenAI model name is required!")

        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_name = model_name
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client."""
        try:
            return AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
        except Exception as e:
            raise ProviderException(f"Failed to initialize OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
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
            response = await self.client.embeddings.create(
                model=self.model_name,
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
