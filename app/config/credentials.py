"""
Azure credential resolution with token broker support.

Provides credential classes and resolution logic:
- TokenBrokerCredential: Fetches tokens from a remote broker API
- BrokerWithFallbackCredential: Tries broker, falls back to local chain
- resolve_credentials(): Picks the right strategy based on TOKEN_BROKER_URL env var
"""

import os
import time
from typing import Any

import requests
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
from azure.core.credentials import TokenCredential, AccessToken
from dotenv import load_dotenv
from loguru import logger

# Load app/.env into os.environ so TOKEN_BROKER_URL (and other vars) are available
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path, override=True)

# Token broker configuration (no default — must be explicitly set via env var)
TOKEN_BROKER_URL = os.getenv("TOKEN_BROKER_URL")

_credentials = None
_broker_credentials = None
_resolved_credentials = None


class TokenBrokerCredential(TokenCredential):
    """
    Azure credential that fetches tokens from a remote token broker API.
    
    The token broker API is expected to accept POST requests with JSON body:
        {"scope": "<scope>"}
    
    And return JSON response:
        {"access_token": "<token>", "expires_on": <unix_timestamp>, "scope": "<scope>"}
    """
    
    def __init__(self, broker_url: str, timeout: int = 30):
        """
        Initialize the token broker credential.
        
        Args:
            broker_url: URL of the token broker /token endpoint
            timeout: Request timeout in seconds
        """
        self._broker_url = broker_url
        self._timeout = timeout
        self._token_cache: dict[str, AccessToken] = {}
    
    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        """
        Fetch an access token from the token broker for the given scope.
        
        Args:
            scopes: The scopes for which to get the token (uses first scope)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            AccessToken with token string and expiration time
            
        Raises:
            Exception: If token broker request fails
        """
        if not scopes:
            raise ValueError("At least one scope is required")
        
        scope = scopes[0]
        
        # Check cache for valid token
        if scope in self._token_cache:
            cached = self._token_cache[scope]
            # Return cached token if it has more than 5 minutes until expiry
            if cached.expires_on > time.time() + 300:
                return cached
        
        # Request new token from broker
        try:
            response = requests.post(
                self._broker_url,
                json={"scope": scope},
                timeout=self._timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            token = AccessToken(
                token=data["access_token"],
                expires_on=int(data["expires_on"])
            )
            
            # Cache the token
            self._token_cache[scope] = token
            logger.debug(f"Token fetched from broker for scope: {scope}")
            
            return token
            
        except requests.RequestException as e:
            logger.error(f"Token broker request failed: {e}")
            raise Exception(f"Failed to get token from broker: {e}") from e
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid token broker response: {e}")
            raise Exception(f"Invalid token broker response: {e}") from e


class BrokerWithFallbackCredential(TokenCredential):
    """Token credential that tries broker first, falls back to local ChainedTokenCredential."""

    def __init__(self, broker: TokenBrokerCredential, fallback: TokenCredential):
        self._broker = broker
        self._fallback = fallback

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        try:
            return self._broker.get_token(*scopes, **kwargs)
        except Exception as e:
            logger.warning(f"Token broker failed ({e}), falling back to ChainedTokenCredential")
            return self._fallback.get_token(*scopes, **kwargs)


def get_credentials():
    """Get or create Azure credentials using local credential chain."""
    global _credentials
    if _credentials is None:
        _credentials = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())
        logger.info("Azure credentials initialized")
    return _credentials


def get_broker_credentials():
    """Get or create Azure credentials using the remote token broker."""
    global _broker_credentials
    if _broker_credentials is None:
        _broker_credentials = TokenBrokerCredential(broker_url=TOKEN_BROKER_URL)
        logger.info(f"Token broker credentials initialized (URL: {TOKEN_BROKER_URL})")
    return _broker_credentials


def resolve_credentials() -> TokenCredential:
    """Resolve credentials based on TOKEN_BROKER_URL environment variable.

    - If TOKEN_BROKER_URL is not set: uses local ChainedTokenCredential.
    - If TOKEN_BROKER_URL is set: uses broker credentials with automatic
      fallback to ChainedTokenCredential on failure.
    """
    global _resolved_credentials
    if _resolved_credentials is not None:
        return _resolved_credentials

    if not os.getenv("TOKEN_BROKER_URL"):
        logger.info("TOKEN_BROKER_URL not set, using local credential chain")
        _resolved_credentials = get_credentials()
    else:
        logger.info("TOKEN_BROKER_URL set, using broker with ChainedTokenCredential fallback")
        _resolved_credentials = BrokerWithFallbackCredential(
            broker=get_broker_credentials(),
            fallback=get_credentials(),
        )
    return _resolved_credentials
