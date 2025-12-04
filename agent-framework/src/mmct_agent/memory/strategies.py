"""Memory strategy implementations."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from mmct_agent.core.types import Message, Role
from mmct_agent.memory.base import BaseMemory, MemoryConfig, InMemoryTokenCounter
from mmct_agent.observability.logging import get_logger

if TYPE_CHECKING:
    from mmct_agent.llm.base import BaseLLMClient

logger = get_logger(__name__)


class SlidingWindowMemory(BaseMemory):
    """Memory strategy that keeps the last N messages.
    
    Simple and efficient strategy for short conversations.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize sliding window memory.
        
        Args:
            window_size: Maximum number of messages to keep.
            config: Additional configuration.
        """
        config = config or MemoryConfig()
        config.window_size = window_size
        super().__init__(config)
        self._token_counter = InMemoryTokenCounter()
    
    @property
    def strategy_name(self) -> str:
        return "sliding_window"
    
    async def add(self, message: Message) -> None:
        """Add a message to memory."""
        self._messages.append(message)
        self._log_operation("add", {"message_role": message.role.value})
    
    async def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages to memory."""
        self._messages.extend(messages)
        self._log_operation("add_many", {"count": len(messages)})
    
    async def get_messages(self) -> list[Message]:
        """Get messages within the window."""
        if len(self._messages) <= self.config.window_size:
            return self._messages.copy()
        return self._messages[-self.config.window_size:]
    
    async def get_context_for_llm(
        self,
        system_message: Message | None = None,
    ) -> list[Message]:
        """Get messages for LLM context."""
        result: list[Message] = []
        
        # Add system message if provided
        if system_message:
            result.append(system_message)
        
        # Get windowed messages
        messages = await self.get_messages()
        
        # Skip any existing system messages in history
        messages = [m for m in messages if m.role != Role.SYSTEM]
        
        result.extend(messages)
        
        self._log_operation(
            "get_context",
            {"total_messages": len(result), "window_applied": len(self._messages) > self.config.window_size},
        )
        
        return result
    
    async def get_token_count(self) -> int:
        """Get token count of current messages."""
        messages = await self.get_messages()
        return self._token_counter.count_messages(messages)


class TokenBasedMemory(BaseMemory):
    """Memory strategy that keeps messages within a token limit.
    
    More precise than sliding window for managing context size.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        token_buffer: int = 500,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize token-based memory.
        
        Args:
            max_tokens: Maximum tokens to keep in context.
            token_buffer: Buffer to reserve for response.
            config: Additional configuration.
        """
        config = config or MemoryConfig()
        config.max_tokens = max_tokens
        config.token_buffer = token_buffer
        super().__init__(config)
        self._token_counter = InMemoryTokenCounter()
    
    @property
    def strategy_name(self) -> str:
        return "token_based"
    
    async def add(self, message: Message) -> None:
        """Add a message to memory."""
        self._messages.append(message)
        self._log_operation("add", {"message_role": message.role.value})
    
    async def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages to memory."""
        self._messages.extend(messages)
        self._log_operation("add_many", {"count": len(messages)})
    
    async def get_messages(self) -> list[Message]:
        """Get messages within token limit."""
        return await self._truncate_to_token_limit(self._messages.copy())
    
    async def get_context_for_llm(
        self,
        system_message: Message | None = None,
    ) -> list[Message]:
        """Get messages for LLM context within token limit."""
        result: list[Message] = []
        available_tokens = self.config.max_tokens - self.config.token_buffer
        
        # Account for system message
        if system_message:
            result.append(system_message)
            available_tokens -= self._token_counter.count_message(system_message)
        
        # Get messages without system messages
        messages = [m for m in self._messages if m.role != Role.SYSTEM]
        
        # Always keep recent messages
        recent_count = self.config.preserve_recent_messages
        recent_messages = messages[-recent_count:] if len(messages) >= recent_count else messages
        older_messages = messages[:-recent_count] if len(messages) > recent_count else []
        
        # Calculate tokens for recent messages
        recent_tokens = self._token_counter.count_messages(recent_messages)
        available_tokens -= recent_tokens
        
        # Add older messages from most recent, until token limit
        kept_older: list[Message] = []
        for msg in reversed(older_messages):
            msg_tokens = self._token_counter.count_message(msg)
            if available_tokens - msg_tokens >= 0:
                kept_older.insert(0, msg)
                available_tokens -= msg_tokens
            else:
                break
        
        result.extend(kept_older)
        result.extend(recent_messages)
        
        self._log_operation(
            "get_context",
            {
                "total_messages": len(result),
                "dropped_messages": len(self._messages) - len(kept_older) - len(recent_messages),
            },
        )
        
        return result
    
    async def get_token_count(self) -> int:
        """Get token count of current messages."""
        return self._token_counter.count_messages(self._messages)
    
    async def _truncate_to_token_limit(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Truncate messages to fit token limit.
        
        Args:
            messages: Messages to truncate.
            
        Returns:
            Truncated message list.
        """
        max_tokens = self.config.max_tokens - self.config.token_buffer
        total_tokens = 0
        result: list[Message] = []
        
        # Process from most recent to oldest
        for msg in reversed(messages):
            msg_tokens = self._token_counter.count_message(msg)
            if total_tokens + msg_tokens <= max_tokens:
                result.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        return result


class SummarizationMemory(BaseMemory):
    """Memory strategy that summarizes older messages using LLM.
    
    Preserves context while dramatically reducing token count.
    Requires an LLM client for summarization.
    """
    
    SUMMARIZATION_PROMPT = """Summarize the following conversation history concisely. 
Focus on key information, decisions made, and important context that would be needed to continue the conversation.
Keep the summary under {max_tokens} tokens.

Conversation:
{conversation}

Summary:"""
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        summarization_threshold: int = 3000,
        summary_max_tokens: int = 500,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize summarization memory.
        
        Args:
            llm_client: LLM client for summarization.
            summarization_threshold: Token count to trigger summarization.
            summary_max_tokens: Maximum tokens for summary.
            config: Additional configuration.
        """
        config = config or MemoryConfig()
        config.summarization_threshold = summarization_threshold
        config.summary_max_tokens = summary_max_tokens
        super().__init__(config)
        
        self._llm_client = llm_client
        self._token_counter = InMemoryTokenCounter()
        self._summary: str | None = None
        self._summarized_until_index: int = 0
    
    @property
    def strategy_name(self) -> str:
        return "summarization"
    
    async def add(self, message: Message) -> None:
        """Add a message and potentially trigger summarization."""
        self._messages.append(message)
        self._log_operation("add", {"message_role": message.role.value})
        
        # Check if summarization is needed
        token_count = await self.get_token_count()
        if token_count > self.config.summarization_threshold:
            await self._summarize_older_messages()
    
    async def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages."""
        for msg in messages:
            await self.add(msg)
    
    async def get_messages(self) -> list[Message]:
        """Get messages with summary if available."""
        messages: list[Message] = []
        
        if self._summary:
            # Add summary as a system message
            messages.append(Message.system(f"Previous conversation summary:\n{self._summary}"))
        
        # Add messages after the summarized portion
        messages.extend(self._messages[self._summarized_until_index:])
        
        return messages
    
    async def get_context_for_llm(
        self,
        system_message: Message | None = None,
    ) -> list[Message]:
        """Get messages for LLM context."""
        result: list[Message] = []
        
        if system_message:
            result.append(system_message)
        
        # Add summary if available
        if self._summary:
            result.append(Message.system(f"Previous conversation summary:\n{self._summary}"))
        
        # Add recent messages (after summary)
        recent_messages = self._messages[self._summarized_until_index:]
        recent_messages = [m for m in recent_messages if m.role != Role.SYSTEM]
        result.extend(recent_messages)
        
        return result
    
    async def get_token_count(self) -> int:
        """Get token count including summary."""
        total = 0
        
        if self._summary:
            total += self._token_counter.count(self._summary) + 20  # Overhead
        
        recent = self._messages[self._summarized_until_index:]
        total += self._token_counter.count_messages(recent)
        
        return total
    
    async def _summarize_older_messages(self) -> None:
        """Summarize older messages to reduce context size."""
        # Keep last N messages unsummarized
        keep_recent = self.config.preserve_recent_messages + 5
        if len(self._messages) <= keep_recent:
            return
        
        # Get messages to summarize
        to_summarize_end = len(self._messages) - keep_recent
        messages_to_summarize = self._messages[self._summarized_until_index:to_summarize_end]
        
        if not messages_to_summarize:
            return
        
        # Build conversation text
        conversation_text = self._format_messages_for_summary(messages_to_summarize)
        
        # Add existing summary if present
        if self._summary:
            conversation_text = f"Previous summary:\n{self._summary}\n\nNew messages:\n{conversation_text}"
        
        # Generate summary
        prompt = self.SUMMARIZATION_PROMPT.format(
            max_tokens=self.config.summary_max_tokens,
            conversation=conversation_text,
        )
        
        try:
            from mmct_agent.core.types import Message as Msg
            response = await self._llm_client.complete(
                messages=[Msg.user(prompt)],
            )
            
            self._summary = response.content
            self._summarized_until_index = to_summarize_end
            
            self._log_operation(
                "summarize",
                {
                    "summarized_count": len(messages_to_summarize),
                    "summary_tokens": self._token_counter.count(self._summary or ""),
                },
            )
            
            logger.debug(f"  📝 Summarized {len(messages_to_summarize)} messages")
            
        except Exception as e:
            logger.error(f"Failed to summarize messages: {e}")
            # Fall back to sliding window behavior
            self._summarized_until_index = to_summarize_end
    
    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        """Format messages for summarization prompt."""
        lines: list[str] = []
        for msg in messages:
            role = msg.role.value.upper()
            content = msg.content or ""
            if msg.tool_calls:
                content += f" [Called tools: {', '.join(tc.name for tc in msg.tool_calls)}]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all messages and summary."""
        super().clear()
        self._summary = None
        self._summarized_until_index = 0


class AdaptiveMemory(BaseMemory):
    """Adaptive memory that switches strategies based on context.
    
    Automatically selects the best strategy based on:
    - Conversation length
    - Token count
    - Message types (tool calls vs regular)
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        max_tokens: int = 4000,
        window_size: int = 20,
        summarization_threshold: int = 3000,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize adaptive memory.
        
        Args:
            llm_client: Optional LLM client for summarization.
            max_tokens: Token limit for context.
            window_size: Sliding window size.
            summarization_threshold: When to trigger summarization.
            config: Additional configuration.
        """
        config = config or MemoryConfig()
        config.max_tokens = max_tokens
        config.window_size = window_size
        config.summarization_threshold = summarization_threshold
        super().__init__(config)
        
        self._llm_client = llm_client
        self._token_counter = InMemoryTokenCounter()
        
        # Initialize sub-strategies
        self._sliding_window = SlidingWindowMemory(window_size, config)
        self._token_based = TokenBasedMemory(max_tokens, config=config)
        self._summarization: SummarizationMemory | None = None
        if llm_client:
            self._summarization = SummarizationMemory(
                llm_client,
                summarization_threshold,
                config=config,
            )
        
        self._current_strategy: str = "token_based"
    
    @property
    def strategy_name(self) -> str:
        return f"adaptive({self._current_strategy})"
    
    async def add(self, message: Message) -> None:
        """Add message and update strategy selection."""
        self._messages.append(message)
        
        # Sync to sub-strategies
        await self._sliding_window.add(message)
        await self._token_based.add(message)
        if self._summarization:
            await self._summarization.add(message)
        
        # Re-evaluate strategy
        await self._select_strategy()
        
        self._log_operation("add", {
            "message_role": message.role.value,
            "current_strategy": self._current_strategy,
        })
    
    async def add_many(self, messages: list[Message]) -> None:
        """Add multiple messages."""
        for msg in messages:
            await self.add(msg)
    
    async def get_messages(self) -> list[Message]:
        """Get messages using current strategy."""
        return await self._get_active_strategy().get_messages()
    
    async def get_context_for_llm(
        self,
        system_message: Message | None = None,
    ) -> list[Message]:
        """Get context using current strategy."""
        return await self._get_active_strategy().get_context_for_llm(system_message)
    
    async def get_token_count(self) -> int:
        """Get token count."""
        return self._token_counter.count_messages(self._messages)
    
    async def _select_strategy(self) -> None:
        """Select the best strategy based on current state."""
        token_count = await self.get_token_count()
        message_count = len(self._messages)
        
        # Count tool-related messages
        tool_message_count = sum(
            1 for m in self._messages
            if m.tool_calls or m.role == Role.TOOL
        )
        tool_ratio = tool_message_count / max(message_count, 1)
        
        old_strategy = self._current_strategy
        
        # Decision logic:
        # 1. If few messages, use sliding window (simple and fast)
        # 2. If many tool calls, use token-based (tool calls have variable length)
        # 3. If high token count and LLM available, use summarization
        # 4. Default to token-based
        
        if message_count <= 10:
            self._current_strategy = "sliding_window"
        elif tool_ratio > 0.3:
            self._current_strategy = "token_based"
        elif token_count > self.config.summarization_threshold and self._summarization:
            self._current_strategy = "summarization"
        else:
            self._current_strategy = "token_based"
        
        if old_strategy != self._current_strategy:
            logger.debug(f"  🧠 Memory: {old_strategy} → {self._current_strategy} ({token_count} tokens)")
    
    def _get_active_strategy(self) -> BaseMemory:
        """Get the currently active strategy."""
        if self._current_strategy == "sliding_window":
            return self._sliding_window
        elif self._current_strategy == "summarization" and self._summarization:
            return self._summarization
        else:
            return self._token_based
    
    def clear(self) -> None:
        """Clear all strategies."""
        super().clear()
        self._sliding_window.clear()
        self._token_based.clear()
        if self._summarization:
            self._summarization.clear()
