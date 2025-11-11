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
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(),override=True)

import os
import time
from typing import Any, Dict, Optional
from dotenv import load_dotenv, find_dotenv

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
        return ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential()
        )
        #return ProxyTokenCredential()

    @staticmethod
    def get_async_credentials():
        """
        Get credentials for Azure services (async version).
        Uses ChainedTokenCredential to try CLI first, then fallback to DefaultAzureCredential.

        Returns:
            AsyncChainedTokenCredential with CLI and DefaultAzureCredential
        """
        return AsyncChainedTokenCredential(
            AsyncAzureCliCredential(),
            AsyncDefaultAzureCredential()
        )
