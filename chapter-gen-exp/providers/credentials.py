"""
Centralized Azure credentials management for all providers.

This module provides a consistent way to obtain Azure credentials across
Azure OpenAI, Azure AI Search, and Azure Blob Storage providers.
"""

from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
from azure.identity.aio import (
    DefaultAzureCredential as AsyncDefaultAzureCredential,
    AzureCliCredential as AsyncAzureCliCredential,
    ChainedTokenCredential as AsyncChainedTokenCredential
)

import os
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv
import requests
from azure.core.credentials import TokenCredential, AccessToken

load_dotenv(find_dotenv(),override=True)


class AzureCredentials:
    """Centralized credential management for all Azure services."""

    @staticmethod
    def get_credentials():
        """
        Get credentials for Azure services (OpenAI, AI Search, Blob Storage).
        Uses ChainedTokenCredential to try CLI first, then fallback to DefaultAzureCredential.

        Returns:
            ChainedTokenCredential with CLI and DefaultAzureCredential
        """
        # return ChainedTokenCredential(
        #     AzureCliCredential(),
        #     DefaultAzureCredential()
        # )
        return ProxyTokenCredential()

    @staticmethod
    def get_async_credentials():
        """
        Get credentials for Azure services (async version).
        Uses ChainedTokenCredential to try CLI first, then fallback to DefaultAzureCredential.

        Returns:
            AsyncChainedTokenCredential with CLI and DefaultAzureCredential
        """
        # return AsyncChainedTokenCredential(
        #     AsyncAzureCliCredential(),
        #     AsyncDefaultAzureCredential()
        # )
        return AsyncProxyTokenCredential()


class ProxyTokenCredential(TokenCredential):
    """Manages Azure credentials for token acquisition with scope-based caching."""
    
    def __init__(self):
        self.api_ep = os.getenv("AZURE_TOKEN_BROKER_EP")
        # Cache to store tokens by scope
        self._token_cache: Dict[str, AccessToken] = {}
        # Fallback credential when api_ep is not available
        self._fallback_credential = AzureCliCredential() if not self.api_ep else None
    
    def get_token(
        self,
        *scopes: str,
        **kwargs: Any,
    ) -> AccessToken:
        """Acquires an access token for the specified scope with caching."""
        scope = scopes[0]
        
        # If no API endpoint is configured, use AzureCliCredential
        if not self.api_ep:
            if self._fallback_credential:
                return self._fallback_credential.get_token(*scopes, **kwargs)
            else:
                raise ValueError("No API endpoint configured and AzureCliCredential not available")
        
        # Check if we have a cached token for this scope
        cached_token = self._get_cached_token(scope)
        if cached_token:
            return cached_token
        
        # Fetch new token if no valid cached token exists
        response = requests.post(
            self.api_ep,
            json={"scope": scope}
        )
        response.raise_for_status()
        
        token_data = response.json()
        access_token = AccessToken(
            token=token_data["access_token"],
            expires_on=token_data["expires_on"]
        )
        
        # Cache the token
        self._token_cache[scope] = access_token
        
        return access_token
    
    def _get_cached_token(self, scope: str) -> Optional[AccessToken]:
        """Get cached token if it exists and is not expired."""
        if scope not in self._token_cache:
            return None
        
        cached_token = self._token_cache[scope]
        current_time = int(time.time())
        
        # Check if token is expired (with 2 hr buffer for safety)
        if cached_token.expires_on <= current_time + 7200:
            # Token is expired or will expire soon, remove from cache
            del self._token_cache[scope]
            return None
        
        return cached_token


class AsyncProxyTokenCredential:
    """Async version: Manages Azure credentials for token acquisition with scope-based caching."""
    
    def __init__(self):
        self.api_ep = os.getenv("AZURE_TOKEN_BROKER_EP")
        # Cache to store tokens by scope
        self._token_cache: Dict[str, AccessToken] = {}
        # Fallback credential when api_ep is not available
        self._fallback_credential = AsyncAzureCliCredential() if not self.api_ep else None
    
    async def get_token(
        self,
        *scopes: str,
        **kwargs: Any,
    ) -> AccessToken:
        """Acquires an access token for the specified scope with caching."""
        scope = scopes[0]
        
        # If no API endpoint is configured, use AsyncAzureCliCredential
        if not self.api_ep:
            if self._fallback_credential:
                return await self._fallback_credential.get_token(*scopes, **kwargs)
            else:
                raise ValueError("No API endpoint configured and AsyncAzureCliCredential not available")
        
        # Check if we have a cached token for this scope
        cached_token = self._get_cached_token(scope)
        if cached_token:
            return cached_token
        
        # Fetch new token if no valid cached token exists
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_ep,
                json={"scope": scope}
            ) as response:
                response.raise_for_status()
                token_data = await response.json()
        
        access_token = AccessToken(
            token=token_data["access_token"],
            expires_on=token_data["expires_on"]
        )
        
        # Cache the token
        self._token_cache[scope] = access_token
        
        return access_token
    
    def _get_cached_token(self, scope: str) -> Optional[AccessToken]:
        """Get cached token if it exists and is not expired."""
        if scope not in self._token_cache:
            return None
        
        cached_token = self._token_cache[scope]
        current_time = int(time.time())
        
        # Check if token is expired (with 2 hr buffer for safety)
        if cached_token.expires_on <= current_time + 7200:
            # Token is expired or will expire soon, remove from cache
            del self._token_cache[scope]
            return None
        
        return cached_token
    
    async def close(self):
        """Close the fallback credential if it exists."""
        if self._fallback_credential:
            await self._fallback_credential.close()
