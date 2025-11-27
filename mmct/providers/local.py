"""Local (custom) provider convenience imports.

This module provides convenient access to all local/custom providers
that don't require cloud services.

Example:
    >>> from mmct.providers.local import LocalStorageProvider, LocalFaissSearchProvider
    >>> storage = LocalStorageProvider(config_dict)
    >>> search = LocalFaissSearchProvider(config_dict)
"""

from .custom_providers.search_provider import CustomSearchProvider
from .custom_providers.local_faiss_search_provider import LocalFaissSearchProvider
from .custom_providers.image_embedding_provider import CustomImageEmbeddingProvider
from .custom_providers.storage_provider import LocalStorageProvider

__all__ = [
    'CustomSearchProvider',
    'LocalFaissSearchProvider',
    'CustomImageEmbeddingProvider',
    'LocalStorageProvider',
]
