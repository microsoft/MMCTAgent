"""Provider system for MMCTAgent."""

from .base import (
    LLMProvider,
    EmbeddingProvider,
    SearchProvider,
    VisionProvider,
    TranscriptionProvider
)
from .factory import ProviderFactory, provider_factory
from .azure_providers import (
    AzureLLMProvider,
    AzureEmbeddingProvider,
    AzureSearchProvider,
    AzureVisionProvider,
    AzureTranscriptionProvider
)
from .openai_providers import (
    OpenAILLMProvider,
    OpenAIEmbeddingProvider,
    OpenAIVisionProvider,
    OpenAITranscriptionProvider
)

__all__ = [
    # Base classes
    'LLMProvider',
    'EmbeddingProvider',
    'SearchProvider',
    'VisionProvider',
    'TranscriptionProvider',
    # Factory
    'ProviderFactory',
    'provider_factory',
    # Azure providers
    'AzureLLMProvider',
    'AzureEmbeddingProvider',
    'AzureSearchProvider',
    'AzureVisionProvider',
    'AzureTranscriptionProvider',
    # OpenAI providers
    'OpenAILLMProvider',
    'OpenAIEmbeddingProvider',
    'OpenAIVisionProvider',
    'OpenAITranscriptionProvider',
]