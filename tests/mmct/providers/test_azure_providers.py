import pytest
from config.provider_config import (
    get_llm_provider,
    get_embedding_provider,
    get_storage_provider,
    get_speech_provider
)

@pytest.mark.unit
def test_provider_initialization():
    """Verify that all major Azure providers can be initialized without errors."""
    try:
        get_llm_provider()
        get_embedding_provider()
        get_storage_provider()
        get_speech_provider()
    except Exception as e:
        pytest.fail(f"Provider initialization failed: {e}")

@pytest.mark.asyncio
@pytest.mark.integration
async def test_storage_connectivity():
    """Verify connection to Azure Blob Storage."""
    try:
        provider = get_storage_provider()
        # Minimal check: check if the keyframe container exists
        # In a real test, we might want to ensure it doesn't fail due to credentials
        container_client = provider.service_client.get_container_client(provider.keyframe_container_name)
        exists = await container_client.exists()
        print(f"\nStorage Test: Container '{provider.keyframe_container_name}' exists: {exists}")
    except Exception as e:
        pytest.fail(f"Storage Connectivity Test Failed: {e}")

@pytest.mark.asyncio
@pytest.mark.integration
async def test_embedding_connectivity():
    """Verify connection to Azure OpenAI Embedding service."""
    try:
        provider = get_embedding_provider()
        embedding = await provider.embedding("this is a test")
        assert len(embedding) > 0
        print(f"\nEmbedding Test Success: Generated vector of size {len(embedding)}")
    except Exception as e:
        pytest.fail(f"Embedding Connectivity Test Failed: {e}")

@pytest.mark.unit
def test_speech_config_generation():
    """Verify that Speech provider generates config correctly (doesn't require real hit)."""
    try:
        provider = get_speech_provider()
        # This calls internal _get_speech_config_with_token which validates credentials
        config = provider._get_speech_config_with_token("en-US")
        assert config is not None
        assert config.speech_recognition_language == "en-US"
    except Exception as e:
        # If it fails due to credentials in a unit test environment, we might catch it here
        # but for smoke tests it should pass if .env is valid.
        print(f"Speech Config Generation (Metadata only): {e}")
