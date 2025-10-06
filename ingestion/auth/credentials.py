"""
Azure authentication utilities with priority order:
1. Azure CLI credentials
2. DefaultAzureCredential (Managed Identity, Environment, etc.)
3. API Key / Connection String (fallback)
"""

import os
import logging
from typing import Optional, Union
from azure.identity import AzureCliCredential, DefaultAzureCredential, ChainedTokenCredential
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


def get_azure_credential(api_key: Optional[str] = None) -> Union[ChainedTokenCredential, AzureKeyCredential]:
    """
    Get Azure credential with prioritized authentication methods.

    Priority order:
    1. Azure CLI credentials (if available)
    2. DefaultAzureCredential (Managed Identity, Environment Variables, etc.)
    3. API Key (if provided)

    Args:
        api_key: Optional API key for fallback authentication

    Returns:
        Azure credential object
    """
    if api_key:
        logger.info("Using API Key authentication")
        return AzureKeyCredential(api_key)

    # Try Azure CLI first, then fall back to DefaultAzureCredential
    try:
        logger.info("Attempting Azure CLI authentication")
        cli_credential = AzureCliCredential()
        # Test if CLI credential works
        cli_credential.get_token("https://management.azure.com/.default")
        logger.info("Successfully authenticated with Azure CLI")
        return cli_credential
    except Exception as cli_error:
        logger.debug(f"Azure CLI authentication not available: {cli_error}")

    # Fall back to DefaultAzureCredential
    try:
        logger.info("Using DefaultAzureCredential (Managed Identity, Environment, etc.)")
        default_credential = DefaultAzureCredential()
        return default_credential
    except Exception as e:
        logger.error(f"Failed to initialize DefaultAzureCredential: {e}")
        raise ValueError(
            "No valid Azure credentials found. Please either:\n"
            "1. Login with Azure CLI: 'az login'\n"
            "2. Set up Managed Identity\n"
            "3. Provide an API key"
        )


def get_storage_credential(connection_string: Optional[str] = None):
    """
    Get Azure Storage credential.

    Priority order:
    1. Connection string (if provided)
    2. Azure CLI credentials
    3. DefaultAzureCredential

    Args:
        connection_string: Optional storage connection string

    Returns:
        Connection string or credential object
    """
    if connection_string:
        logger.info("Using Storage connection string")
        return connection_string

    # For blob storage without connection string, use chained credential
    try:
        logger.info("Using Azure credential for Storage (CLI -> Default)")
        credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential()
        )
        return credential
    except Exception as e:
        logger.error(f"Failed to get storage credential: {e}")
        raise ValueError(
            "No valid Azure Storage credentials found. Please either:\n"
            "1. Login with Azure CLI: 'az login'\n"
            "2. Set AZURE_STORAGE_CONNECTION_STRING environment variable\n"
            "3. Provide a connection string"
        )


def test_credential(credential, scope: str = "https://management.azure.com/.default") -> bool:
    """
    Test if a credential is valid by attempting to get a token.

    Args:
        credential: Azure credential to test
        scope: OAuth scope to test against

    Returns:
        True if credential is valid, False otherwise
    """
    try:
        if isinstance(credential, AzureKeyCredential):
            # API keys don't use tokens
            return True

        token = credential.get_token(scope)
        return token is not None
    except Exception as e:
        logger.debug(f"Credential test failed: {e}")
        return False
