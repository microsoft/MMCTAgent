"""Base interface for Large Language Model (LLM) providers.

This module defines the abstract interface that all LLM providers (e.g., Azure, 
OpenAI) must implement to be compatible with the MMCT system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class and 
    implement its abstract methods to ensure consistent integration with 
    the rest of the MMCT ecosystem.
    """

    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Generates a chat completion response from a list of messages.

        Args:
            messages: A list of message dictionaries (e.g., {"role": "user", 
                "content": "..."}).
            **kwargs: Additional provider-specific arguments (e.g., temperature, 
                max_tokens).

        Returns:
            Dict[str, Any]: The completion response from the LLM.
        """
        pass

    @abstractmethod
    def get_autogen_client(self, **kwargs: Any) -> Any:
        """Returns an AutoGen-compatible chat completion client.

        Args:
            **kwargs: Configuration arguments for the AutoGen client.

        Returns:
            Any: An instance of a chat completion client that adheres to the 
                AutoGen interface.
        """
        pass

    @abstractmethod
    async def generate_json(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Generates a structured JSON response from a text prompt.

        This is a convenience method that simplifies structured data 
        generation by wrapping chat completion with JSON enforcement.

        Args:
            prompt: The instruction or query for the LLM.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Dict[str, Any]: The parsed JSON response as a dictionary.
        """
        pass
