"""Custom exceptions for the agent framework."""

from __future__ import annotations

from typing import Any


class AgentFrameworkError(Exception):
    """Base exception for all agent framework errors."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LLMError(AgentFrameworkError):
    """Error related to LLM API calls."""
    
    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        status_code: int | None = None,
        retry_count: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code
        self.retry_count = retry_count


class ToolExecutionError(AgentFrameworkError):
    """Error during tool execution."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        original_error: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.original_error = original_error


class HandoffError(AgentFrameworkError):
    """Error during agent handoff."""
    
    def __init__(
        self,
        message: str,
        source_agent: str,
        target_agent: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.source_agent = source_agent
        self.target_agent = target_agent


class MemoryError(AgentFrameworkError):
    """Error related to memory management."""
    
    def __init__(
        self,
        message: str,
        memory_type: str = "unknown",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.memory_type = memory_type


class ConfigurationError(AgentFrameworkError):
    """Error in configuration or settings."""
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.config_key = config_key


class MaxIterationsError(AgentFrameworkError):
    """Error when maximum iterations are exceeded."""
    
    def __init__(
        self,
        message: str,
        iterations: int,
        max_iterations: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.iterations = iterations
        self.max_iterations = max_iterations


class ToolNotFoundError(AgentFrameworkError):
    """Error when a requested tool is not found."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        available_tools: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class ToolTimeoutError(ToolExecutionError):
    """Error when tool execution times out."""
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        timeout_seconds: float,
        arguments: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, tool_name, arguments, None, details)
        self.timeout_seconds = timeout_seconds
