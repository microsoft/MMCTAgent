import pytest
from unittest.mock import patch, MagicMock
from config.provider_config import get_settings, get_llm_provider
from mmct.providers.azure_providers import AzureLLMProvider

@pytest.mark.unit
def test_settings_loading(monkeypatch):
    """Test that settings are correctly loaded from environment variables."""
    monkeypatch.setenv("LLM_ENDPOINT", "https://unit-test.openai.azure.com/")
    monkeypatch.setenv("LLM_API_KEY", "unit-test-key")
    
    # Reload settings for this test
    # Note: get_settings is lru_cached, so we need to clear it or bypass it
    from config.provider_config import ProviderEnvSettings
    settings = ProviderEnvSettings()
    
    assert settings.llm_endpoint == "https://unit-test.openai.azure.com/"
    assert settings.llm_api_key == "unit-test-key"

@pytest.mark.unit
def test_placeholder_ignoring(monkeypatch):
    """Verify that placeholder API keys are treated as not provided."""
    monkeypatch.setenv("LLM_API_KEY", "<your-api-key>")
    
    with patch("config.provider_config.resolve_credentials") as mock_resolve:
        # Returning a Mock instead of a string to satisfy get_bearer_token_provider
        mock_resolve.return_value = MagicMock()
        
        # We need to ensure get_settings returns our mocked env
        from config.provider_config import ProviderEnvSettings
        mock_settings = ProviderEnvSettings(llm_api_key="<your-api-key>", llm_endpoint="http://test")
        
        with patch("config.provider_config.get_settings", return_value=mock_settings):
            provider = get_llm_provider()
            # If logic is correct, it should have called resolve_credentials 
            # instead of using "<your-api-key>"
            mock_resolve.assert_called_once()

@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.smoke
async def test_llm_connectivity_smoke():
    """Perform a minimal real LLM hit to test accessibility.
    
    This test will only pass if a valid .env is present with real credentials.
    """
    try:
        provider = get_llm_provider()
        messages = [{"role": "user", "content": "hi"}]
        response = await provider.chat_completion(messages=messages, max_tokens=5)
        
        assert response is not None
        assert "content" in response
        print(f"\nSmoke Test Success: Received response from LLM")
    except Exception as e:
        pytest.fail(f"LLM Connectivity Smoke Test Failed: {e}")
