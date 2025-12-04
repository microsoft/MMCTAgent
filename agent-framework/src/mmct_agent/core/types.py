"""Core type definitions for the agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator
from uuid import uuid4


class Role(str, Enum):
    """Message role in conversation."""
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM."""
    
    id: str
    name: str
    arguments: dict[str, Any]
    
    @classmethod
    def from_openai(cls, tool_call: Any) -> ToolCall:
        """Create ToolCall from OpenAI API response."""
        import json
        return cls(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )


@dataclass
class ToolResult:
    """Result of a tool execution."""
    
    tool_call_id: str
    name: str
    result: Any
    error: str | None = None
    execution_time_ms: float = 0.0
    
    @property
    def is_success(self) -> bool:
        """Check if tool execution was successful."""
        return self.error is None
    
    def to_content(self) -> str:
        """Convert result to string content for LLM."""
        if self.error:
            return f"Error: {self.error}"
        if isinstance(self.result, str):
            return self.result
        import json
        try:
            return json.dumps(self.result, indent=2, default=str)
        except (TypeError, ValueError):
            return str(self.result)


@dataclass
class Message:
    """A message in the conversation history."""
    
    role: Role
    content: str | None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    
    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI API message format."""
        msg: dict[str, Any] = {"role": self.role.value}
        
        if self.content is not None:
            msg["content"] = self.content
            
        if self.name and self.role == Role.TOOL:
            msg["name"] = self.name
            
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            tc.arguments if isinstance(tc.arguments, str) 
                            else __import__("json").dumps(tc.arguments)
                        ),
                    },
                }
                for tc in self.tool_calls
            ]
            
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
            
        return msg
    
    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content)
    
    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)
    
    @classmethod
    def tool_result(cls, tool_result: ToolResult) -> Message:
        """Create a tool result message."""
        return cls(
            role=Role.TOOL,
            content=tool_result.to_content(),
            name=tool_result.name,
            tool_call_id=tool_result.tool_call_id,
        )


@dataclass
class TokenUsage:
    """Token usage statistics."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two token usages together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class AgentResponse:
    """Response from an agent execution."""
    
    content: str | None
    messages: list[Message]
    tool_results: list[ToolResult] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    agent_name: str = ""
    handoff_to: str | None = None
    handoff_context: dict[str, Any] | None = None
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    
    @property
    def has_handoff(self) -> bool:
        """Check if response includes a handoff request."""
        return self.handoff_to is not None


@dataclass
class StreamChunk:
    """A chunk of streaming response."""
    
    content: str
    is_complete: bool = False
    tool_calls: list[ToolCall] | None = None
    token_usage: TokenUsage | None = None


@dataclass
class HandoffRequest:
    """Request to hand off to another agent."""
    
    target_agent: str
    context: dict[str, Any] = field(default_factory=dict)  # Structured content from schema
