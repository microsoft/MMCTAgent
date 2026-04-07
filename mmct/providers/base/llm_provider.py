from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate chat completion response."""
        pass

    async def generate_json(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Generate a JSON-parsed response via chat_completion.

        Convenience wrapper that calls chat_completion with json_object response
        format and returns the parsed dict. Subclasses may override for stricter
        schema enforcement.
        """
        import json
        response = await self.chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        content = response.get("content", "{}")
        if isinstance(content, str):
            return json.loads(content)
        # Already parsed (e.g. structured output path returned a dict/model)
        if hasattr(content, "model_dump"):
            return content.model_dump()
        return content

    @abstractmethod
    def get_autogen_client(self, **kwargs):
        """Get autogen-compatible client for the LLM provider."""
        pass
