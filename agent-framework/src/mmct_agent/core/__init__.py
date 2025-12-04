"""Core module - agents, swarm, types, and exceptions."""

from mmct_agent.core.agent import Agent, AgentHooks, get_default_agent_hooks
from mmct_agent.core.swarm import Swarm, SwarmHooks, get_default_swarm_hooks
from mmct_agent.core.types import Message, Role, AgentResponse, ToolResult, ToolCall
from mmct_agent.core.exceptions import (
    AgentFrameworkError,
    LLMError,
    ToolExecutionError,
    HandoffError,
    MemoryError,
    ConfigurationError,
    MaxIterationsError,
)

__all__ = [
    "Agent",
    "AgentHooks",
    "get_default_agent_hooks",
    "Swarm",
    "SwarmHooks",
    "get_default_swarm_hooks",
    "Message",
    "Role",
    "AgentResponse",
    "ToolResult",
    "ToolCall",
    "AgentFrameworkError",
    "LLMError",
    "ToolExecutionError",
    "HandoffError",
    "MemoryError",
    "ConfigurationError",
    "MaxIterationsError",
]
