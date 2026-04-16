"""Azure OpenAI text embedding provider implementation.

This module provides the AzureEmbeddingProvider class, which implements the 
BaseEmbeddingProvider interface for generating vector embeddings via 
Azure OpenAI services.
"""

from typing import List, Union, Optional, Any
from azure.identity import get_bearer_token_provider
from loguru import logger
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from openai import AsyncAzureOpenAI

from mmct.providers.base import BaseEmbeddingProvider
from mmct.utils.error_handler import ProviderException, ConfigurationException, handle_exceptions, convert_exceptions


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider implementation.

    This provider handles authentication and client management for 
    generating text embeddings using Azure OpenAI's embedding models. 
    It supports both single-string and batch embedding generation.

    Attributes:
        endpoint (str): The Azure OpenAI service endpoint.
        deployment_name (str): The name of the specific embedding deployment.
        api_version (str): The Azure OpenAI API version.
        credentials (Union[AzureKeyCredential, AsyncTokenCredential], optional): 
            Identity-based credentials.
        api_key (str, optional): Key-based authentication string.
        timeout (int): Request timeout in seconds.
        max_retries (int): Maximum retry attempts.
        client (AsyncAzureOpenAI): The initialized async client.
    """

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
        """Initializes the AzureEmbeddingProvider.

        Args:
            endpoint: Azure OpenAI endpoint URL.
            deployment_name: Name of the embedding deployment.
            api_version: Azure OpenAI API version.
            credentials: Azure credentials for token-based authentication.
                Mutually exclusive with `api_key`.
            api_key: API key for key-based authentication.
                Mutually exclusive with `credentials`.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.

        Raises:
            ConfigurationException: If required fields are missing or if both
                `credentials` and `api_key` are provided.
        """
        if not endpoint:
            raise ConfigurationException("Azure OpenAI endpoint is required for Embedding Provider!")

        if not deployment_name:
            raise ConfigurationException("Azure OpenAI deployment name is required for Embedding Provider!")
        
        if not api_version:
            raise ConfigurationException("Azure OpenAI api version is required for Embedding Provider!")

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
    
    def _initialize_client(self) -> AsyncAzureOpenAI:
        """Initializes the Azure OpenAI client with either credentials or API key.

        Returns:
            AsyncAzureOpenAI: The initialized asynchronous client.

        Raises:
            ProviderException: If client initialization fails.
        """
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
    async def embedding(self, text: str, **kwargs: Any) -> List[float]:
        """Generates an embedding for a specific text string using Azure OpenAI.

        Args:
            text: The input text to be vectorised.
            **kwargs: Additional parameters passed to the OpenAI embeddings API.

        Returns:
            List[float]: The normalized vector embedding.

        Raises:
            ProviderException: If the embedding request fails.
        """
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
    async def batch_embedding(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generates vector embeddings for a batch of text strings.

        Args:
            texts: A list of input strings to be vectorised.
            **kwargs: Additional parameters passed to the OpenAI embeddings API.

        Returns:
            List[List[float]]: A list of normalized vector embeddings.

        Raises:
            ProviderException: If the batch embedding request fails.
        """
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

    def get_async_client(self) -> AsyncAzureOpenAI:
        """Returns the underlying async Azure OpenAI client.

        Returns:
            AsyncAzureOpenAI: The active async client.
        """
        return self.client

    async def close(self) -> None:
        """Closes the embedding client and releases underlying resources."""
        if self.client:
            logger.info("Closing Azure OpenAI embedding client")
            await self.client.close()