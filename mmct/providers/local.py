"""Local (custom) provider convenience imports.

This module provides convenient access to all local/custom providers
that don't require cloud services.

Example:
    >>> from mmct.providers.local import LocalStorageProvider, LocalFaissSearchProvider
    >>> storage = LocalStorageProvider(config_dict)
    >>> search = LocalFaissSearchProvider(config_dict)
"""

from .custom_providers.graph_rag_search_provider import GraphRagSearchProvider
from .custom_providers.local_faiss_search_provider import LocalFaissSearchProvider
from .custom_providers.clip_image_embedding_provider import ClipImageEmbeddingProvider
from .custom_providers.local_storage_provider import LocalStorageProvider

__all__ = [
    'GraphRagSearchProvider',
    'LocalFaissSearchProvider',
    'ClipImageEmbeddingProvider',
    'LocalStorageProvider',
]
