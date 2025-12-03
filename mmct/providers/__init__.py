"""Provider system for MMCTAgent.

Import base classes for custom providers:
    from mmct.providers.base import BaseLLMProvider, BaseEmbeddingProvider

Import concrete providers directly from convenience modules:
    from mmct.providers.azure import AzureLLMProvider
    from mmct.providers.openai import OpenAILLMProvider
    from mmct.providers.local import LocalFaissSearchProvider
"""

from .base import (
    BaseLLMProvider,
    BaseEmbeddingProvider,
    BaseChapterVectorDBProvider,
    BaseKeyframesVectorDBProvider,
    BaseObjectCollectionVectorDBProvider,
    BaseVisionProvider,
    BaseTranscriptionProvider,
    BaseStorageProvider,
    BaseImageEmbeddingProvider
)

__all__ = [
    'BaseLLMProvider',
    'BaseEmbeddingProvider',
    'BaseChapterVectorDBProvider',
    'BaseKeyframesVectorDBProvider',
    'BaseObjectCollectionVectorDBProvider',
    'BaseVisionProvider',
    'BaseTranscriptionProvider',
    'BaseStorageProvider',
    'BaseImageEmbeddingProvider',
]