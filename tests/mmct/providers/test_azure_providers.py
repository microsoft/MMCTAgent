"""
Azure Provider Connectivity Tests
===================================
Real integration checks for Azure service providers.
All tests require a valid .env with real credentials.

These tests are a subset of tests/connectivity/test_service_connectivity.py,
focused specifically on the provider abstraction layer.
"""

import pytest
from config.provider_config import (
    get_settings,
    get_embedding_provider,
    get_storage_provider,
    get_speech_provider,
)


def _skip_if_placeholder(value: str, env_var: str) -> None:
    if not value or str(value).startswith("<your"):
        pytest.skip(f"{env_var} not configured in .env")


@pytest.mark.asyncio
@pytest.mark.connectivity
async def test_storage_connectivity():
    """Verify connection to Azure Blob Storage."""
    s = get_settings()
    _skip_if_placeholder(s.storage_account_name, "STORAGE_ACCOUNT_NAME")

    provider = get_storage_provider()
    container_client = provider.service_client.get_container_client(
        provider.keyframe_container_name
    )
    exists = await container_client.exists()
    print(f"\nStorage OK — container '{provider.keyframe_container_name}' exists={exists}")


@pytest.mark.asyncio
@pytest.mark.connectivity
async def test_embedding_connectivity():
    """Verify connection to Azure OpenAI Embedding service."""
    s = get_settings()
    _skip_if_placeholder(s.embedding_service_endpoint, "EMBEDDING_SERVICE_ENDPOINT")

    provider = get_embedding_provider()
    vector = await provider.embedding("connectivity check")

    assert len(vector) > 0, "Embedding returned empty vector"
    print(f"\nEmbedding OK — vector size={len(vector)}")


@pytest.mark.connectivity
def test_speech_connectivity():
    """Verify Azure Speech Service token exchange."""
    s = get_settings()
    _skip_if_placeholder(s.speech_service_region, "SPEECH_SERVICE_REGION")

    provider = get_speech_provider()
    config = provider._get_speech_config_with_token("en-US")

    assert config is not None
    assert config.speech_recognition_language == "en-US"
    print(f"\nSpeech OK — region={s.speech_service_region}")
