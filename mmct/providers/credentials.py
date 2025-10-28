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
from loguru import logger


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
        logger.info("Creating chained credential (CLI -> DefaultAzureCredential)")
        return ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential()
        )

    @staticmethod
    def get_async_credentials():
        """
        Get credentials for Azure services (async version).
        Uses ChainedTokenCredential to try CLI first, then fallback to DefaultAzureCredential.

        Returns:
            AsyncChainedTokenCredential with CLI and DefaultAzureCredential
        """
        logger.info("Creating async chained credential (CLI -> DefaultAzureCredential)")
        return AsyncChainedTokenCredential(
            AsyncAzureCliCredential(),
            AsyncDefaultAzureCredential()
        )
