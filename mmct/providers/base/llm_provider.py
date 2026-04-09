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

    @abstractmethod
    async def generate_json(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a JSON response from a prompt.
        
        This is a convenience method that wraps chat_completion with JSON output format.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional arguments passed to chat_completion
            
        Returns:
            Parsed JSON response as a dictionary
        """
        pass
