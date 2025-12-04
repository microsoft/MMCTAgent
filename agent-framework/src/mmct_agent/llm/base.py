"""Base LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from mmct_agent.core.types import Message, StreamChunk, TokenUsage, ToolCall
    from mmct_agent.tools.base import ToolDefinition


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    
    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM completion."""
    
    content: str | None
    tool_calls: list[ToolCall] | None
    token_usage: TokenUsage
    finish_reason: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    raw_response: Any = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.
    
    All LLM provider implementations must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize the LLM client.
        
        Args:
            config: LLM configuration options.
        """
        self.config = config or LLMConfig()
        self._total_token_usage: TokenUsage | None = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the LLM provider."""
        ...
    
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given messages.
        
        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions for function calling.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            LLMResponse with the completion result.
            
        Raises:
            LLMError: If the API call fails after retries.
        """
        ...
    
    @abstractmethod
    async def complete_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion for the given messages.
        
        Args:
            messages: List of conversation messages.
            tools: Optional list of tool definitions for function calling.
            **kwargs: Additional provider-specific parameters.
            
        Yields:
            StreamChunk objects with partial content.
            
        Raises:
            LLMError: If the API call fails after retries.
        """
        ...
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            Number of tokens.
        """
        ...
    
    @abstractmethod
    async def count_messages_tokens(self, messages: list[Message]) -> int:
        """Count the total tokens in a list of messages.
        
        Args:
            messages: List of messages to count tokens for.
            
        Returns:
            Total number of tokens.
        """
        ...
    
    def get_total_token_usage(self) -> TokenUsage:
        """Get the total token usage across all calls.
        
        Returns:
            Cumulative token usage.
        """
        from mmct_agent.core.types import TokenUsage
        return self._total_token_usage or TokenUsage()
    
    def reset_token_usage(self) -> None:
        """Reset the cumulative token usage counter."""
        self._total_token_usage = None
    
    def _update_token_usage(self, usage: TokenUsage) -> None:
        """Update cumulative token usage.
        
        Args:
            usage: Token usage from a single call.
        """
        from mmct_agent.core.types import TokenUsage
        if self._total_token_usage is None:
            self._total_token_usage = TokenUsage()
        self._total_token_usage = self._total_token_usage + usage
