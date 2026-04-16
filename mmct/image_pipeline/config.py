"""Provider configuration for the image pipeline agents."""
from pydantic import BaseModel, Field, ConfigDict
from mmct.providers.base import BaseLLMProvider


class ImageAgentProviderConfig(BaseModel):
    """Provider config for V4ImageAgent — bundles the LLM provider needed for vision tools."""

    llm_provider: BaseLLMProvider = Field(...)

    model_config = ConfigDict(arbitrary_types_allowed=True)