import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from config.provider_config import ProviderEnvSettings
from mmct.providers.base import BaseLLMProvider

@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_settings():
    """Returns a ProviderEnvSettings object with mocked/test values."""
    return ProviderEnvSettings(
        llm_endpoint="https://test.openai.azure.com/",
        llm_deployment_name="test-gpt",
        llm_model_name="gpt-4o",
        llm_api_key="test-key",
        embedding_service_endpoint="https://test.openai.azure.com/",
        embedding_service_deployment_name="test-embed",
        storage_account_name="teststorage",
        keyframe_container_name="test-keyframes",
        speech_service_region="eastus",
        speech_service_resource_id="/subscriptions/test/resourceGroups/test/providers/Microsoft.CognitiveServices/accounts/test"
    )

@pytest.fixture(autouse=True)
def clear_provider_caches():
    """Clear LRU caches for all provider factory functions before each test."""
    from config import provider_config
    provider_config.get_settings.cache_clear()
    provider_config.get_llm_provider.cache_clear()
    provider_config.get_reasoning_llm_provider.cache_clear()
    provider_config.get_embedding_provider.cache_clear()
    provider_config.get_storage_provider.cache_clear()
    provider_config.get_speech_provider.cache_clear()
    provider_config.get_whisper_provider.cache_clear()
    yield

@pytest.fixture
def mock_llm_provider():
    """Fixture for a mocked BaseLLMProvider."""
    # Using a spec allows it to pass isinstance(obj, BaseLLMProvider) checks in Pydantic
    mock = MagicMock(spec=BaseLLMProvider)
    mock.chat_completion = AsyncMock(return_value={
        "content": "Test response",
        "usage": {"total_tokens": 10},
        "model": "gpt-4o",
        "finish_reason": "stop"
    })
    return mock
