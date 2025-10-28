
from mmct.providers.base import EmbeddingProvider
from typing import Dict, Any, List
from azure.identity import get_bearer_token_provider
from loguru import logger
from mmct.utils.error_handler import ProviderException, ConfigurationException
from openai import AsyncAzureOpenAI, AzureOpenAI
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.providers.credentials import AzureCredentials


class AzureEmbeddingProvider(EmbeddingProvider):
    """Azure OpenAI embedding provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_async_credentials()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client."""
        try:
            endpoint = self.config.get("endpoint")
            api_version = self.config.get("api_version", "2024-08-01-preview")
            use_managed_identity = self.config.get("use_managed_identity", True)
            timeout = self.config.get("timeout", 200)
            max_retries = self.config.get("max_retries", 2)
            
            if not endpoint:
                raise ConfigurationException("Azure OpenAI endpoint is required")
            
            if use_managed_identity:
                token_provider = get_bearer_token_provider(
                    self.credential, 
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=max_retries,
                    timeout=timeout
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure OpenAI API key is required when managed identity is disabled")
                
                return AsyncAzureOpenAI(
                    api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    max_retries=max_retries,
                    timeout=timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            deployment_name = self.config.get("deployment_name") or self.config.get("embedding_deployment_name")
            if not deployment_name:
                raise ConfigurationException(
                    "Azure OpenAI embedding deployment name is required. "
                    "Set EMBEDDING_SERVICE_DEPLOYMENT_NAME environment variable."
                )
            
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=text,
                **kwargs
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            raise ProviderException(f"Azure OpenAI embedding failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def batch_embedding(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings for multiple texts using Azure OpenAI."""
        try:
            deployment_name = self.config.get("deployment_name") or self.config.get("embedding_deployment_name")
            if not deployment_name:
                raise ConfigurationException(
                    "Azure OpenAI embedding deployment name is required. "
                    "Set EMBEDDING_SERVICE_DEPLOYMENT_NAME environment variable."
                )
            
            response = await self.client.embeddings.create(
                model=deployment_name,
                input=texts,
                **kwargs
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Azure OpenAI batch embedding failed: {e}")
            raise ProviderException(f"Azure OpenAI batch embedding failed: {e}")