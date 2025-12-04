"""Metrics collection for agents and swarms."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolMetrics:
    """Metrics for a single tool."""
    
    name: str
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_execution_time_ms: float = 0.0
    
    @property
    def avg_execution_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time_ms / self.call_count
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    
    agent_name: str
    run_count: int = 0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    handoffs_initiated: int = 0
    handoffs_received: int = 0
    errors: int = 0
    tool_metrics: dict[str, ToolMetrics] = field(default_factory=dict)
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.run_count == 0:
            return 0.0
        return self.total_latency_ms / self.run_count
    
    @property
    def avg_tokens_per_run(self) -> float:
        """Calculate average tokens per run."""
        if self.run_count == 0:
            return 0.0
        return self.total_tokens / self.run_count
    
    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
    ) -> None:
        """Record a tool call.
        
        Args:
            tool_name: Name of the tool.
            success: Whether the call succeeded.
            execution_time_ms: Execution time in milliseconds.
        """
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics(name=tool_name)
        
        metrics = self.tool_metrics[tool_name]
        metrics.call_count += 1
        metrics.total_execution_time_ms += execution_time_ms
        
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1
        
        self.tool_calls += 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "run_count": self.run_count,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_run": self.avg_tokens_per_run,
            "tool_calls": self.tool_calls,
            "handoffs_initiated": self.handoffs_initiated,
            "handoffs_received": self.handoffs_received,
            "errors": self.errors,
            "tools": {
                name: {
                    "call_count": m.call_count,
                    "success_rate": m.success_rate,
                    "avg_execution_time_ms": m.avg_execution_time_ms,
                }
                for name, m in self.tool_metrics.items()
            },
        }


@dataclass
class SwarmMetrics:
    """Metrics for a swarm execution."""
    
    swarm_id: str
    run_count: int = 0
    total_latency_ms: float = 0.0
    total_iterations: int = 0
    total_handoffs: int = 0
    total_tokens: int = 0
    agents_used: set[str] = field(default_factory=set)
    success_count: int = 0
    error_count: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.run_count == 0:
            return 0.0
        return self.total_latency_ms / self.run_count
    
    @property
    def avg_iterations_per_run(self) -> float:
        """Calculate average iterations per run."""
        if self.run_count == 0:
            return 0.0
        return self.total_iterations / self.run_count
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.run_count == 0:
            return 0.0
        return self.success_count / self.run_count
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "swarm_id": self.swarm_id,
            "run_count": self.run_count,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "total_iterations": self.total_iterations,
            "avg_iterations_per_run": self.avg_iterations_per_run,
            "total_handoffs": self.total_handoffs,
            "total_tokens": self.total_tokens,
            "agents_used": list(self.agents_used),
            "success_rate": self.success_rate,
            "success_count": self.success_count,
            "error_count": self.error_count,
        }


class MetricsCollector:
    """Collects and aggregates metrics from agents and swarms."""
    
    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._swarm_metrics: dict[str, SwarmMetrics] = {}
        self._start_time = datetime.utcnow()
    
    def get_agent_metrics(self, agent_name: str) -> AgentMetrics:
        """Get or create metrics for an agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            AgentMetrics instance.
        """
        if agent_name not in self._agent_metrics:
            self._agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        return self._agent_metrics[agent_name]
    
    def get_swarm_metrics(self, swarm_id: str) -> SwarmMetrics:
        """Get or create metrics for a swarm.
        
        Args:
            swarm_id: ID of the swarm.
            
        Returns:
            SwarmMetrics instance.
        """
        if swarm_id not in self._swarm_metrics:
            self._swarm_metrics[swarm_id] = SwarmMetrics(swarm_id=swarm_id)
        return self._swarm_metrics[swarm_id]
    
    def record_agent_run(
        self,
        agent_name: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        tool_calls: int = 0,
        error: bool = False,
    ) -> None:
        """Record an agent run.
        
        Args:
            agent_name: Name of the agent.
            latency_ms: Run latency in milliseconds.
            prompt_tokens: Prompt tokens used.
            completion_tokens: Completion tokens used.
            tool_calls: Number of tool calls made.
            error: Whether the run resulted in an error.
        """
        metrics = self.get_agent_metrics(agent_name)
        metrics.run_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.prompt_tokens += prompt_tokens
        metrics.completion_tokens += completion_tokens
        metrics.total_tokens += prompt_tokens + completion_tokens
        metrics.tool_calls += tool_calls
        
        if error:
            metrics.errors += 1
    
    def record_handoff(self, from_agent: str, to_agent: str) -> None:
        """Record a handoff between agents.
        
        Args:
            from_agent: Source agent name.
            to_agent: Target agent name.
        """
        from_metrics = self.get_agent_metrics(from_agent)
        from_metrics.handoffs_initiated += 1
        
        to_metrics = self.get_agent_metrics(to_agent)
        to_metrics.handoffs_received += 1
    
    def record_swarm_run(
        self,
        swarm_id: str,
        latency_ms: float,
        iterations: int,
        handoffs: int,
        total_tokens: int,
        agents_used: list[str],
        success: bool,
    ) -> None:
        """Record a swarm run.
        
        Args:
            swarm_id: ID of the swarm.
            latency_ms: Total latency in milliseconds.
            iterations: Number of iterations.
            handoffs: Number of handoffs.
            total_tokens: Total tokens used.
            agents_used: List of agents used.
            success: Whether the run succeeded.
        """
        metrics = self.get_swarm_metrics(swarm_id)
        metrics.run_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.total_iterations += iterations
        metrics.total_handoffs += handoffs
        metrics.total_tokens += total_tokens
        metrics.agents_used.update(agents_used)
        
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1
    
    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics.
        
        Returns:
            Dictionary with all metrics.
        """
        return {
            "collection_start": self._start_time.isoformat(),
            "collection_duration_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "agents": {
                name: m.to_dict()
                for name, m in self._agent_metrics.items()
            },
            "swarms": {
                id: m.to_dict()
                for id, m in self._swarm_metrics.items()
            },
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._agent_metrics.clear()
        self._swarm_metrics.clear()
        self._start_time = datetime.utcnow()


# Global metrics collector
_global_collector: MetricsCollector | None = None


def get_global_collector() -> MetricsCollector:
    """Get the global metrics collector.
    
    Returns:
        Global MetricsCollector instance.
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
