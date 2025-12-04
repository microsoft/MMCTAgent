"""Azure provider convenience imports.

This module provides convenient access to all Azure-based providers.

Example:
    >>> from mmct.providers.azure import AzureLLMProvider, AzureSearchProvider
    >>> llm = AzureLLMProvider(config_dict)
    >>> search = AzureSearchProvider(config_dict)
"""

from .azure_providers.llm_provider import AzureLLMProvider
from .azure_providers.embedding_provider import AzureEmbeddingProvider
from .azure_providers.ai_search_chapter_provider import AISearchChapterProvider
from .azure_providers.ai_search_object_collection_provider import AISearchObjectCollectionProvider
from .azure_providers.ai_search_keyframes_provider import AISearchKeyframesProvider
from .azure_providers.storage_provider import AzureStorageProvider
from .azure_providers.vision_provider import AzureVisionProvider
from .azure_providers.whisper_transcription_provider import WhisperTranscriptionProvider
from .azure_providers.speech_service_provider import AzureSpeechServiceProvider

__all__ = [
    'AzureLLMProvider',
    'AzureEmbeddingProvider',
    'AISearchChapterProvider',
    'AISearchObjectCollectionProvider',
    'AISearchKeyframesProvider',
    'AzureStorageProvider',
    'AzureVisionProvider',
    'WhisperTranscriptionProvider',
    'AzureSpeechServiceProvider',
]
