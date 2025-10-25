import os
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv
import requests
from azure.core.credentials import TokenCredential, AccessToken
from azure.identity import AzureCliCredential

load_dotenv(find_dotenv(),override=True)

class CustomTokenCredential(TokenCredential):
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