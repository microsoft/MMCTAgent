"""Swarm-level memory for sharing context between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime

from mmct_agent.core.types import Message, Role
from mmct_agent.memory.base import BaseMemory, MemoryConfig, InMemoryTokenCounter
from mmct_agent.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentContext:
    """Context information from an agent's execution."""
    
    agent_name: str
    summary: str | None = None
    key_findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_usage: int = 0


@dataclass
class SwarmState:
    """State of the swarm execution."""
    
    current_agent: str | None = None
    visited_agents: list[str] = field(default_factory=list)
    handoff_count: int = 0
    total_iterations: int = 0
    agent_contexts: dict[str, AgentContext] = field(default_factory=dict)
    shared_data: dict[str, Any] = field(default_factory=dict)


class SwarmMemory:
    """Shared memory for swarm-level context.
    
    Maintains:
    - Summaries from each agent's work
    - Shared data between agents
    - Handoff history
    - Aggregated context for routing decisions
    """
    
    def __init__(
        self,
        max_context_tokens: int = 2000,
        preserve_agent_count: int = 5,
    ) -> None:
        """Initialize swarm memory.
        
        Args:
            max_context_tokens: Maximum tokens for shared context.
            preserve_agent_count: Number of recent agent contexts to keep.
        """
        self._max_context_tokens = max_context_tokens
        self._preserve_agent_count = preserve_agent_count
        self._token_counter = InMemoryTokenCounter()
        
        self._state = SwarmState()
        self._operation_log: list[dict[str, Any]] = []
    
    @property
    def state(self) -> SwarmState:
        """Get current swarm state."""
        return self._state
    
    def set_current_agent(self, agent_name: str) -> None:
        """Set the currently active agent.
        
        Args:
            agent_name: Name of the active agent.
        """
        self._state.current_agent = agent_name
        if agent_name not in self._state.visited_agents:
            self._state.visited_agents.append(agent_name)
        self._log_operation("set_current_agent", {"agent": agent_name})
    
    def record_handoff(
        self,
        from_agent: str,
        to_agent: str,
        context: dict[str, Any] | None = None,
        summary: str | None = None,
    ) -> None:
        """Record a handoff between agents.
        
        Args:
            from_agent: Source agent name.
            to_agent: Target agent name.
            context: Context passed in handoff.
            summary: Summary of work done.
        """
        self._state.handoff_count += 1
        
        # Store context from source agent
        if from_agent:
            agent_ctx = self._state.agent_contexts.get(from_agent, AgentContext(agent_name=from_agent))
            if summary:
                agent_ctx.summary = summary
            if context:
                agent_ctx.metadata.update(context)
            self._state.agent_contexts[from_agent] = agent_ctx
        
        self._log_operation("handoff", {
            "from": from_agent,
            "to": to_agent,
            "has_context": context is not None,
            "has_summary": summary is not None,
        })
        
        logger.debug(f"Recorded handoff: {from_agent} -> {to_agent}")
    
    def add_agent_context(
        self,
        agent_name: str,
        summary: str | None = None,
        key_findings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update context from an agent.
        
        Args:
            agent_name: Agent name.
            summary: Summary of agent's work.
            key_findings: Key findings/results.
            metadata: Additional metadata.
        """
        ctx = self._state.agent_contexts.get(agent_name, AgentContext(agent_name=agent_name))
        
        if summary:
            ctx.summary = summary
            ctx.token_usage = self._token_counter.count(summary)
        if key_findings:
            ctx.key_findings.extend(key_findings)
        if metadata:
            ctx.metadata.update(metadata)
        ctx.timestamp = datetime.utcnow()
        
        self._state.agent_contexts[agent_name] = ctx
        
        # Prune old contexts if needed
        self._prune_old_contexts()
        
        self._log_operation("add_agent_context", {"agent": agent_name})
    
    def get_context_for_agent(
        self,
        agent_name: str,
        include_all_agents: bool = False,
    ) -> str:
        """Get formatted context for an agent.
        
        This returns only the swarm state overview (agents involved, handoff count).
        Agent-specific content is passed via structured handoff schemas, not through
        this context method.
        
        Args:
            agent_name: Target agent name.
            include_all_agents: Whether to include context from all agents (unused).
            
        Returns:
            Formatted context string with swarm state overview.
        """
        lines: list[str] = []
        
        # Add swarm state overview only
        if self._state.visited_agents:
            lines.append(f"Agents involved: {', '.join(self._state.visited_agents)}")
            lines.append(f"Handoffs so far: {self._state.handoff_count}")
        
        return "\n".join(lines)
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data by key.
        
        Args:
            key: Data key.
            default: Default value if not found.
            
        Returns:
            Stored value or default.
        """
        return self._state.shared_data.get(key, default)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """Set shared data.
        
        Args:
            key: Data key.
            value: Value to store.
        """
        self._state.shared_data[key] = value
        self._log_operation("set_shared_data", {"key": key})
    
    def increment_iteration(self) -> int:
        """Increment and return iteration count.
        
        Returns:
            New iteration count.
        """
        self._state.total_iterations += 1
        return self._state.total_iterations
    
    def _prune_old_contexts(self) -> None:
        """Remove old agent contexts to stay within limits."""
        if len(self._state.agent_contexts) <= self._preserve_agent_count:
            return
        
        # Sort by timestamp and keep most recent
        sorted_agents = sorted(
            self._state.agent_contexts.items(),
            key=lambda x: x[1].timestamp,
            reverse=True,
        )
        
        self._state.agent_contexts = dict(sorted_agents[:self._preserve_agent_count])
    
    def clear(self) -> None:
        """Clear all swarm state."""
        self._state = SwarmState()
        self._log_operation("clear", {})
    
    def get_operation_log(self) -> list[dict[str, Any]]:
        """Get operation log for debugging.
        
        Returns:
            List of operations.
        """
        return self._operation_log.copy()
    
    def _log_operation(self, operation: str, details: dict[str, Any]) -> None:
        """Log an operation."""
        self._operation_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            **details,
        })
