"""Authentication utilities for Azure services."""

from .credentials import get_azure_credential, get_storage_credential

__all__ = [
    "get_azure_credential",
    "get_storage_credential",
]
