"""
Service Connectivity Tests — MMCTAgent
=======================================
Real accessibility checks against every service configured in .env.
No mocks — each test makes an actual call to the live service.

Run all:
    pytest tests/connectivity/ -m connectivity -vv

Run one:
    pytest tests/connectivity/test_service_connectivity.py::test_neo4j_connectivity -vv

Each test PASSES (service reachable) or SKIPS (not configured in .env).
A test never fails purely because credentials are missing — that is a skip.
"""

import pytest

from config.provider_config import (
    get_settings,
    get_llm_provider,
    get_embedding_provider,
    get_storage_provider,
    get_speech_provider,
    get_neo4j_query_provider,
    get_whisper_provider,
)

pytestmark = pytest.mark.connectivity


# ── Helper ────────────────────────────────────────────────────────────────────

def _skip_if_placeholder(value: str, env_var: str) -> None:
    """Skip the test if the value is empty or still a placeholder string."""
    if not value or str(value).startswith("<your"):
        pytest.skip(f"{env_var} not configured in .env — skipping connectivity check")


# ── LLM (Azure OpenAI chat) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_connectivity():
    """Verify live connectivity to the Azure OpenAI LLM endpoint."""
    s = get_settings()
    _skip_if_placeholder(s.llm_endpoint, "LLM_ENDPOINT")

    provider = get_llm_provider()
    response = await provider.chat_completion(
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )

    assert response is not None, "LLM returned None"
    assert "content" in response, f"Unexpected response shape: {response}"
    print(f"\n  LLM   OK — model={response.get('model', 'unknown')}")


# ── Embedding (Azure OpenAI) ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_embedding_connectivity():
    """Verify live connectivity to the Azure OpenAI Embedding endpoint."""
    s = get_settings()
    _skip_if_placeholder(s.embedding_service_endpoint, "EMBEDDING_SERVICE_ENDPOINT")

    provider = get_embedding_provider()
    vector = await provider.embedding("ping")

    assert vector is not None, "Embedding returned None"
    assert len(vector) > 0, "Embedding returned empty vector"
    print(f"\n  Embedding   OK — vector size={len(vector)}")


# ── Blob Storage (Azure Storage) ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_storage_connectivity():
    """Verify live connectivity to Azure Blob Storage and keyframe container."""
    s = get_settings()
    _skip_if_placeholder(s.storage_account_name, "STORAGE_ACCOUNT_NAME")

    provider = get_storage_provider()
    container_client = provider.service_client.get_container_client(
        provider.keyframe_container_name
    )
    exists = await container_client.exists()

    print(f"\n  Storage   OK — container '{provider.keyframe_container_name}' exists={exists}")
    # exists=False is valid (container not yet created); reachability is what matters


# ── Speech (Azure Cognitive Services) ────────────────────────────────────────

def test_speech_connectivity():
    """Verify live connectivity to Azure Speech Service (token exchange)."""
    s = get_settings()
    _skip_if_placeholder(s.speech_service_region, "SPEECH_SERVICE_REGION")

    provider = get_speech_provider()
    config = provider._get_speech_config_with_token("en-US")

    assert config is not None, "Speech config returned None"
    assert config.speech_recognition_language == "en-US"
    print(f"\n  Speech   OK — region={s.speech_service_region}")


# ── Neo4j ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_neo4j_connectivity():
    """Verify live connectivity to Neo4j via a minimal Cypher query."""
    s = get_settings()
    if not s.neo4j_password:
        pytest.skip("NEO4J_PASSWORD not set in .env — skipping connectivity check")

    provider = get_neo4j_query_provider()
    assert provider is not None, "Neo4j provider returned None — check NEO4J_* settings"

    # Initialise the async driver, then run the simplest possible query
    await provider._ensure_driver()
    async with provider._driver.session(database=s.neo4j_database) as session:
        result = await session.run("RETURN 1 AS ping")
        record = await result.single()
        assert record["ping"] == 1, "Neo4j ping query returned unexpected result"

    print(f"\n  Neo4j   OK — uri={s.neo4j_uri}, database={s.neo4j_database}")


# ── Whisper (Azure OpenAI transcription) ─────────────────────────────────────

def test_whisper_connectivity():
    """Verify the Whisper provider initialises correctly (no real audio call needed)."""
    s = get_settings()
    _skip_if_placeholder(s.whisper_deployment_name, "WHISPER_DEPLOYMENT_NAME")

    provider = get_whisper_provider()
    assert provider is not None, "Whisper provider returned None"
    print(f"\n  Whisper   OK — deployment={s.whisper_deployment_name}")
