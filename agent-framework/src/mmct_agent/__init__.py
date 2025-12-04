"""MMCT Agent Framework - A blazingly fast, async-first agentic framework."""

from mmct_agent.core.agent import Agent, AgentHooks
from mmct_agent.core.swarm import Swarm, SwarmConfig, SwarmResult, SwarmHooks
from mmct_agent.core.types import Message, Role, AgentResponse, ToolResult, ToolCall, TokenUsage
from mmct_agent.core.exceptions import (
    AgentFrameworkError,
    LLMError,
    ToolExecutionError,
    HandoffError,
    MaxIterationsError,
)
from mmct_agent.tools.decorator import tool
from mmct_agent.tools.registry import ToolRegistry
from mmct_agent.tools.base import ToolDefinition

__version__ = "0.1.0"

__all__ = [
    # Core Agent
    "Agent",
    "AgentHooks",
    # Swarm
    "Swarm",
    "SwarmConfig",
    "SwarmResult",
    "SwarmHooks",
    # Types
    "Message",
    "Role",
    "AgentResponse",
    "ToolResult",
    "ToolCall",
    "TokenUsage",
    # Exceptions
    "AgentFrameworkError",
    "LLMError",
    "ToolExecutionError",
    "HandoffError",
    "MaxIterationsError",
    # Tools
    "tool",
    "ToolRegistry",
    "ToolDefinition",
]
