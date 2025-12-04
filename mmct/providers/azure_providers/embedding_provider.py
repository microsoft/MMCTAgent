
from mmct.providers.base import BaseEmbeddingProvider
from typing import List, Union, Optional
from azure.identity import get_bearer_token_provider
from loguru import logger
from mmct.utils.error_handler import ProviderException, ConfigurationException
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from openai import AsyncAzureOpenAI
from mmct.utils.error_handler import handle_exceptions, convert_exceptions


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider implementation."""

    def __init__(
        self,
        endpoint: str,
        deployment_name: str,
        api_version: str = "2024-08-01-preview",
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        api_key: Optional[str] = None,
        timeout: int = 200,
        max_retries: int = 2
    ):
        """Initialize AzureEmbeddingProvider.

        Args:
            endpoint: Azure OpenAI endpoint URL
            deployment_name: Name of the embedding deployment
            api_version: Azure OpenAI API version (default: 2024-08-01-preview)
            credentials: Azure credentials for token-based authentication (mutually exclusive with api_key)
            api_key: API key for key-based authentication (mutually exclusive with credentials)
            timeout: Request timeout in seconds (default: 200)
            max_retries: Maximum number of retry attempts (default: 2)

        Raises:
            ConfigurationException: If neither credentials nor api_key is provided,
                                   or if both are provided, or if required fields are missing
        """
        if not endpoint:
            raise ConfigurationException("Azure OpenAI endpoint is required for Embedding Provider!")

        if not deployment_name:
            raise ConfigurationException("Azure OpenAI deployment name is required for Embedding Provider!")
        
        if not api_version:
            raise ConfigurationException("Azure OpenAI api version is required for Whisper Transcription Provider!")

        # Validate that exactly one of credentials or api_key is provided
        if credentials is None and api_key is None:
            raise ConfigurationException("Either credentials or api_key must be provided!")

        if credentials is not None and api_key is not None:
            raise ConfigurationException("Only one of credentials or api_key should be provided, not both!")

        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.credentials = credentials
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client with either credentials or API key."""
        try:
            if self.credentials is not None:
                # Use credentials with token-based authentication
                token_provider = get_bearer_token_provider(
                    self.credentials,
                    "https://cognitiveservices.azure.com/.default"
                )
                return AsyncAzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    azure_ad_token_provider=token_provider,
                    max_retries=self.max_retries,
                    timeout=self.timeout
                )
            else:
                # Use API key authentication
                return AsyncAzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                    api_key=self.api_key,
                    max_retries=self.max_retries,
                    timeout=self.timeout
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure OpenAI client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def embedding(self, text: str, **kwargs) -> List[float]:
        """Generate embedding using Azure OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.deployment_name,
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
            response = await self.client.embeddings.create(
                model=self.deployment_name,
                input=texts,
                **kwargs
            )

            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Azure OpenAI batch embedding failed: {e}")
            raise ProviderException(f"Azure OpenAI batch embedding failed: {e}")

    def get_async_client(self):
        """Get async OpenAI client for direct embeddings API access."""
        return self.client

    async def close(self):
        """Close the embedding client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure OpenAI embedding client")
            await self.client.close()