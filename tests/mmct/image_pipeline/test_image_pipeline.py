import pytest
from mmct.image_pipeline.config import ImageAgentProviderConfig
from mmct.providers.azure_providers import AzureLLMProvider

@pytest.mark.unit
def test_image_agent_config_initialization(mock_llm_provider):
    """Test that ImageAgentProviderConfig properly accepts an LLM provider."""
    try:
        config = ImageAgentProviderConfig(llm_provider=mock_llm_provider)
        assert config.llm_provider == mock_llm_provider
    except Exception as e:
        pytest.fail(f"ImageAgentProviderConfig initialization failed: {e}")

@pytest.mark.unit
def test_image_agent_provider_from_config():
    """Verify get_image_agent_provider from main config."""
    from config.provider_config import get_image_agent_provider
    try:
        config = get_image_agent_provider()
        assert isinstance(config, ImageAgentProviderConfig)
        assert isinstance(config.llm_provider, AzureLLMProvider)
    except Exception as e:
        pytest.fail(f"get_image_agent_provider failed: {e}")
