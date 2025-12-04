from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion response."""
        pass

    @abstractmethod
    def get_autogen_client(self, **kwargs):
        """Get autogen-compatible client for the LLM provider."""
        pass
