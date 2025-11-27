"""Azure provider convenience imports.

This module provides convenient access to all Azure-based providers.

Example:
    >>> from mmct.providers.azure import AzureLLMProvider, AzureSearchProvider
    >>> llm = AzureLLMProvider(config_dict)
    >>> search = AzureSearchProvider(config_dict)
"""

from .azure_providers.llm_provider import AzureLLMProvider
from .azure_providers.embedding_provider import AzureEmbeddingProvider
from .azure_providers.search_provider import AzureSearchProvider
from .azure_providers.storage_provider import AzureStorageProvider
from .azure_providers.vision_provider import AzureVisionProvider
from .azure_providers.whisper_transcription_provider import WhisperTranscriptionProvider
from .azure_providers.speech_service_provider import AzureSpeechServiceProvider

__all__ = [
    'AzureLLMProvider',
    'AzureEmbeddingProvider',
    'AzureSearchProvider',
    'AzureStorageProvider',
    'AzureVisionProvider',
    'WhisperTranscriptionProvider',
    'AzureSpeechServiceProvider',
]
