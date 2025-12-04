"""Base memory abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mmct_agent.core.types import Message
    from mmct_agent.llm.base import BaseLLMClient


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    # Common settings
    preserve_system_message: bool = True
    preserve_recent_messages: int = 2  # Always keep last N messages
    
    # Token-based settings
    max_tokens: int = 4000
    token_buffer: int = 500  # Reserve for response
    
    # Sliding window settings
    window_size: int = 20
    
    # Summarization settings
    summarization_threshold: int = 3000  # Token count to trigger summarization
    summary_max_tokens: int = 500
    
    # Adaptive settings
    strategy: str = "adaptive"  # "sliding_window", "token_based", "summarization", "adaptive"
    
    # Extra parameters for extensions
    extra: dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """Abstract base class for memory management strategies.
    
    Memory strategies control how conversation history is managed
    to prevent context length explosion while preserving relevant information.
    """
    
    def __init__(self, config: MemoryConfig | None = None) -> None:
        """Initialize memory with configuration.
        
        Args:
            config: Memory configuration options.
        """
        self.config = config or MemoryConfig()
        self._messages: list[Message] = []
        self._operation_log: list[dict[str, Any]] = []  # For debugging
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the name of the memory strategy."""
        ...
    
    @abstractmethod
    async def add(self, message: Message) -> None:
        """Add a message to memory.
        
        Args:
            message: Message to add.
        """
        ...
    
    @abstractmethod
    async def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages to memory.
        
        Args:
            messages: Messages to add.
        """
        ...
    
    @abstractmethod
    async def get_messages(self) -> list[Message]:
        """Get all messages in memory, after applying strategy.
        
        Returns:
            List of messages within context limits.
        """
        ...
    
    @abstractmethod
    async def get_context_for_llm(
        self,
        system_message: Message | None = None,
    ) -> list[Message]:
        """Get messages formatted for LLM context.
        
        This applies the memory strategy and ensures the context
        fits within token limits.
        
        Args:
            system_message: Optional system message to prepend.
            
        Returns:
            List of messages ready for LLM.
        """
        ...
    
    @abstractmethod
    async def get_token_count(self) -> int:
        """Get the current token count of messages in memory.
        
        Returns:
            Approximate token count.
        """
        ...
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        self._log_operation("clear", {"message_count": len(self._messages)})
        self._messages.clear()
    
    def get_raw_messages(self) -> list[Message]:
        """Get all messages without applying strategy.
        
        Returns:
            All stored messages.
        """
        return self._messages.copy()
    
    def get_operation_log(self) -> list[dict[str, Any]]:
        """Get the log of memory operations for debugging.
        
        Returns:
            List of operation records.
        """
        return self._operation_log.copy()
    
    def clear_operation_log(self) -> None:
        """Clear the operation log."""
        self._operation_log.clear()
    
    def _log_operation(self, operation: str, details: dict[str, Any]) -> None:
        """Log a memory operation.
        
        Args:
            operation: Name of the operation.
            details: Operation details.
        """
        from datetime import datetime
        self._operation_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "strategy": self.strategy_name,
            **details,
        })
    
    def __len__(self) -> int:
        """Return number of messages in memory."""
        return len(self._messages)


class InMemoryTokenCounter:
    """Simple token counter without LLM client dependency."""
    
    def __init__(self) -> None:
        """Initialize token counter."""
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._encoding = None
    
    def count(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Token count (approximate if tiktoken not available).
        """
        if self._encoding:
            return len(self._encoding.encode(text))
        # Fallback: rough approximation
        return len(text) // 4
    
    def count_message(self, message: Message) -> int:
        """Count tokens in a message.
        
        Args:
            message: Message to count tokens for.
            
        Returns:
            Token count.
        """
        total = 4  # Base tokens for role, separators
        if message.content:
            total += self.count(message.content)
        if message.name:
            total += self.count(message.name)
        if message.tool_calls:
            import json
            for tc in message.tool_calls:
                total += self.count(tc.name)
                total += self.count(json.dumps(tc.arguments))
        return total
    
    def count_messages(self, messages: list[Message]) -> int:
        """Count tokens in multiple messages.
        
        Args:
            messages: Messages to count.
            
        Returns:
            Total token count.
        """
        return sum(self.count_message(m) for m in messages) + 3  # Priming
