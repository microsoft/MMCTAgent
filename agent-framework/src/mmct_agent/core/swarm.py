"""Swarm orchestration for multi-agent collaboration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING
from uuid import uuid4

from mmct_agent.core.types import Message, AgentResponse, TokenUsage, Role
from mmct_agent.core.exceptions import (
    MaxIterationsError,
    HandoffError,
    AgentFrameworkError,
)
from mmct_agent.core.agent import Agent
from mmct_agent.memory.swarm_memory import SwarmMemory
from mmct_agent.observability.logging import set_trace_id, clear_trace_id, get_logger

logger = get_logger(__name__)


# Callback types for swarm events
OnAgentStartCallback = Callable[[str, str], Awaitable[None] | None]  # agent_name, trace_id
OnAgentCompleteCallback = Callable[[str, AgentResponse], Awaitable[None] | None]
OnHandoffCallback = Callable[[str, str, dict[str, Any]], Awaitable[None] | None]  # from, to, context
OnSwarmCompleteCallback = Callable[["SwarmResult"], Awaitable[None] | None]


@dataclass
class SwarmHooks:
    """Hooks for observing swarm behavior."""
    
    on_agent_start: OnAgentStartCallback | None = None
    on_agent_complete: OnAgentCompleteCallback | None = None
    on_handoff: OnHandoffCallback | None = None
    on_swarm_complete: OnSwarmCompleteCallback | None = None
    
    @classmethod
    def with_default_logging(cls) -> "SwarmHooks":
        """Create hooks with default debug logging for all events."""
        
        def log_agent_start(agent_name: str, trace_id: str) -> None:
            logger.info(f"{'─' * 40}")
            logger.info(f"▶ Agent: {agent_name}")
        
        def log_agent_complete(agent_name: str, response: AgentResponse) -> None:
            tokens = response.token_usage.total_tokens if response.token_usage else 0
            latency = response.latency_ms
            if response.content:
                status = "completed"
            elif response.has_handoff:
                status = f"handing off → {response.handoff_to}"
            else:
                status = "no response"
            logger.info(f"◀ Agent {agent_name} {status} ({tokens} tokens, {latency:.0f}ms)")
        
        def log_handoff(from_agent: str, to_agent: str, context: dict[str, Any]) -> None:
            reason = context.get("reason", "")[:80] if context else ""
            logger.info(f"🔀 Handoff: {from_agent} → {to_agent}")
            if reason:
                logger.debug(f"   Reason: {reason}...")
        
        def log_swarm_complete(result: "SwarmResult") -> None:
            logger.info(f"{'═' * 40}")
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            agents = " → ".join(result.agents_used)
            tokens = result.total_token_usage.total_tokens if result.total_token_usage else 0
            logger.info(
                f"Swarm {status} | Agents: {agents} | "
                f"Iterations: {result.iterations} | Tokens: {tokens} | "
                f"Time: {result.total_latency_ms:.0f}ms"
            )
        
        return cls(
            on_agent_start=log_agent_start,
            on_agent_complete=log_agent_complete,
            on_handoff=log_handoff,
            on_swarm_complete=log_swarm_complete,
        )


def get_default_swarm_hooks() -> SwarmHooks:
    """Get default swarm hooks with debug logging enabled."""
    return SwarmHooks.with_default_logging()


@dataclass
class SwarmConfig:
    """Configuration for swarm execution."""
    
    max_iterations: int = 20
    max_agent_iterations: int = 5  # Max times same agent can run consecutively
    parallel_agents: bool = False  # Allow parallel agent execution
    timeout_seconds: float = 300.0  # Total swarm timeout
    shared_memory_enabled: bool = True


@dataclass
class SwarmResult:
    """Result from swarm execution."""
    
    final_response: AgentResponse | None
    agent_responses: list[AgentResponse]
    total_token_usage: TokenUsage
    total_latency_ms: float
    iterations: int
    agents_used: list[str]
    trace_id: str
    success: bool = True
    error: str | None = None


class Swarm:
    """Orchestrator for multi-agent collaboration.
    
    Manages:
    - Agent handoffs with context transformation
    - Shared swarm memory
    - Iteration limits and safety
    - Observability and persistence
    """
    
    def __init__(
        self,
        agents: list[Agent] | dict[str, Agent],
        config: SwarmConfig | None = None,
        hooks: SwarmHooks | None = None,
        swarm_memory: SwarmMemory | None = None,
    ) -> None:
        """Initialize a swarm.
        
        Args:
            agents: List or dict of agents in the swarm.
            config: Swarm configuration.
            hooks: Callback hooks.
            swarm_memory: Shared swarm memory.
        """
        self.config = config or SwarmConfig()
        self.hooks = hooks or get_default_swarm_hooks()
        
        # Convert list to dict if needed
        if isinstance(agents, list):
            self._agents = {agent.name: agent for agent in agents}
        else:
            self._agents = agents
        
        # Initialize shared memory
        self._swarm_memory = swarm_memory or SwarmMemory()
    
    @property
    def agents(self) -> dict[str, Agent]:
        """Get all agents in the swarm."""
        return self._agents
    
    @property
    def swarm_memory(self) -> SwarmMemory:
        """Get the shared swarm memory."""
        return self._swarm_memory
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the swarm.
        
        Args:
            agent: Agent to add.
        """
        if agent.name in self._agents:
            logger.warning(f"Replacing existing agent: {agent.name}")
        self._agents[agent.name] = agent
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the swarm.
        
        Args:
            name: Agent name.
            
        Returns:
            True if removed, False if not found.
        """
        if name in self._agents:
            del self._agents[name]
            return True
        return False
    
    def get_agent(self, name: str) -> Agent | None:
        """Get an agent by name.
        
        Args:
            name: Agent name.
            
        Returns:
            Agent or None.
        """
        return self._agents.get(name)
    
    async def run(
        self,
        initial_agent: str,
        task: str | Message,
        trace_id: str | None = None,
        initial_context: dict[str, Any] | None = None,
    ) -> SwarmResult:
        """Run the swarm starting with a specific agent.
        
        Args:
            initial_agent: Name of the starting agent.
            task: Initial task or message.
            trace_id: Optional trace ID.
            initial_context: Initial context for the swarm.
            
        Returns:
            SwarmResult with all responses and metrics.
        """
        trace_id = trace_id or str(uuid4())
        set_trace_id(trace_id)  # Set trace_id for all subsequent logs
        start_time = time.perf_counter()
        
        logger.info(f"{'═' * 40}")
        logger.info(f"Swarm starting | Initial agent: {initial_agent} | Trace: {trace_id[:8]}...")
        
        # Validate initial agent
        if initial_agent not in self._agents:
            raise HandoffError(
                message=f"Initial agent '{initial_agent}' not found in swarm",
                source_agent="swarm",
                target_agent=initial_agent,
            )
        
        # Reset swarm memory
        self._swarm_memory.clear()
        
        # Set initial context
        if initial_context:
            for key, value in initial_context.items():
                self._swarm_memory.set_shared_data(key, value)
        
        # Convert task to message if needed
        if isinstance(task, str):
            task = Message.user(task)
        
        # Track execution
        agent_responses: list[AgentResponse] = []
        total_tokens = TokenUsage()
        agents_used: list[str] = []
        
        current_agent_name = initial_agent
        current_message = task
        consecutive_runs: dict[str, int] = {}
        
        try:
            iteration = 0
            
            while iteration < self.config.max_iterations:
                iteration += 1
                self._swarm_memory.increment_iteration()
                
                # Check consecutive runs
                consecutive_runs[current_agent_name] = consecutive_runs.get(current_agent_name, 0) + 1
                if consecutive_runs[current_agent_name] > self.config.max_agent_iterations:
                    raise MaxIterationsError(
                        message=f"Agent {current_agent_name} exceeded max consecutive iterations",
                        iterations=consecutive_runs[current_agent_name],
                        max_iterations=self.config.max_agent_iterations,
                    )
                
                # Get current agent
                current_agent = self._agents[current_agent_name]
                self._swarm_memory.set_current_agent(current_agent_name)
                
                if current_agent_name not in agents_used:
                    agents_used.append(current_agent_name)
                
                # Trigger start hook
                await self._trigger_hook(
                    self.hooks.on_agent_start,
                    current_agent_name,
                    trace_id,
                )
                
                # Prepare message with swarm context
                if self.config.shared_memory_enabled and iteration > 1:
                    swarm_context = self._swarm_memory.get_context_for_agent(current_agent_name)
                    if swarm_context:
                        # Prepend swarm context to the message
                        if current_message.content:
                            enhanced_content = f"[Swarm Context]\n{swarm_context}\n\n[Current Task]\n{current_message.content}"
                            current_message = Message.user(enhanced_content)
                
                # Run the agent
                response = await current_agent.run(current_message, trace_id)
                
                agent_responses.append(response)
                total_tokens = total_tokens + response.token_usage
                
                # Trigger complete hook
                await self._trigger_hook(
                    self.hooks.on_agent_complete,
                    current_agent_name,
                    response,
                )
                
                # Store agent context in swarm memory
                self._swarm_memory.add_agent_context(
                    agent_name=current_agent_name,
                    summary=response.content[:500] if response.content else None,
                    metadata={"token_usage": response.token_usage.total_tokens},
                )
                
                # Check for handoff
                if response.has_handoff:
                    next_agent = response.handoff_to
                    
                    if next_agent not in self._agents:
                        raise HandoffError(
                            message=f"Handoff target '{next_agent}' not found in swarm",
                            source_agent=current_agent_name,
                            target_agent=next_agent,
                        )
                    
                    # Trigger handoff hook
                    await self._trigger_hook(
                        self.hooks.on_handoff,
                        current_agent_name,
                        next_agent,
                        response.handoff_context or {},
                    )
                    
                    # Record handoff in swarm memory
                    self._swarm_memory.record_handoff(
                        from_agent=current_agent_name,
                        to_agent=next_agent,
                        context=response.handoff_context,
                        summary=response.handoff_context.get("summary") if response.handoff_context else None,
                    )
                    
                    # Prepare context for next agent
                    next_message = await self._prepare_handoff_message(
                        current_agent,
                        response,
                    )
                    
                    # Reset consecutive count for previous agent
                    consecutive_runs = {next_agent: 0}
                    
                    current_agent_name = next_agent
                    current_message = next_message
                else:
                    # No handoff - swarm is complete
                    break
            
            if iteration >= self.config.max_iterations:
                raise MaxIterationsError(
                    message="Swarm exceeded maximum iterations",
                    iterations=iteration,
                    max_iterations=self.config.max_iterations,
                )
            
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            
            result = SwarmResult(
                final_response=agent_responses[-1] if agent_responses else None,
                agent_responses=agent_responses,
                total_token_usage=total_tokens,
                total_latency_ms=total_latency_ms,
                iterations=iteration,
                agents_used=agents_used,
                trace_id=trace_id,
                success=True,
            )
            
            # Trigger complete hook
            await self._trigger_hook(self.hooks.on_swarm_complete, result)
            
            return result
            
        except AgentFrameworkError as e:
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            
            result = SwarmResult(
                final_response=agent_responses[-1] if agent_responses else None,
                agent_responses=agent_responses,
                total_token_usage=total_tokens,
                total_latency_ms=total_latency_ms,
                iterations=self._swarm_memory.state.total_iterations,
                agents_used=agents_used,
                trace_id=trace_id,
                success=False,
                error=str(e),
            )
            
            logger.error(
                f"Swarm failed: {e}",
                extra={"trace_id": trace_id},
                exc_info=True,
            )
            
            raise
        finally:
            clear_trace_id()  # Clear trace_id when swarm completes
    
    async def run_parallel(
        self,
        agents: list[str],
        task: str | Message,
        trace_id: str | None = None,
    ) -> list[AgentResponse]:
        """Run multiple agents in parallel on the same task.
        
        Args:
            agents: List of agent names to run.
            task: Task for all agents.
            trace_id: Optional trace ID.
            
        Returns:
            List of responses from all agents.
        """
        trace_id = trace_id or str(uuid4())
        
        # Validate agents
        for agent_name in agents:
            if agent_name not in self._agents:
                raise HandoffError(
                    message=f"Agent '{agent_name}' not found in swarm",
                    source_agent="swarm",
                    target_agent=agent_name,
                )
        
        # Convert task to message if needed
        if isinstance(task, str):
            task = Message.user(task)
        
        # Run agents in parallel
        async def run_agent(agent_name: str) -> AgentResponse:
            agent = self._agents[agent_name]
            return await agent.run(task, trace_id)
        
        tasks = [run_agent(name) for name in agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results: list[AgentResponse] = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Agent {agents[i]} failed: {response}")
                # Create error response
                results.append(AgentResponse(
                    content=f"Error: {response}",
                    messages=[],
                    agent_name=agents[i],
                    trace_id=trace_id,
                ))
            else:
                results.append(response)
        
        return results
    
    async def _prepare_handoff_message(
        self,
        from_agent: Agent,
        response: AgentResponse,
    ) -> Message:
        """Prepare the message for handoff to the next agent.
        
        The handoff context contains structured data as defined by the
        content_schema in register_handoff. This is passed directly to
        the target agent.
        
        Args:
            from_agent: Agent that initiated the handoff.
            response: Response from the source agent.
            
        Returns:
            Message for the next agent.
        """
        import json
        
        context = response.handoff_context or {}
        
        # Format the structured handoff content as the message
        # The schema ensures the LLM provides the required fields
        if context:
            handoff_content = json.dumps(context, indent=2)
        else:
            handoff_content = response.content or ""
        
        return Message.user(handoff_content)
    
    async def save_memory(
        self,
        path: str = "./swarm_logs",
        session_id: str | None = None,
        include_agents: bool = True,
    ) -> str:
        """Save swarm and agent memories to disk for debugging.
        
        Args:
            path: Base directory for memory logs.
            session_id: Optional session identifier. Auto-generated if not provided.
            include_agents: Whether to also save individual agent memories.
            
        Returns:
            Path where memory was saved.
        """
        from mmct_agent.memory.persistence import MemoryPersistence
        
        persistence = MemoryPersistence(base_path=path, session_id=session_id)
        
        # Save agent memories if requested
        agent_memories = {}
        if include_agents:
            agent_memories = {
                name: agent.memory
                for name, agent in self._agents.items()
            }
        
        await persistence.save_all(agent_memories, self._swarm_memory)
        
        logger.info(f"Swarm memory saved to {persistence._session_path}")
        return str(persistence._session_path)
    
    async def _trigger_hook(self, hook: Callable | None, *args: Any) -> None:
        """Trigger a hook if defined.
        
        Args:
            hook: Hook callback.
            *args: Arguments to pass.
        """
        if hook is None:
            return
        
        try:
            result = hook(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.warning(f"Swarm hook raised exception: {e}", exc_info=True)
    
    def reset(self) -> None:
        """Reset all agents and swarm memory."""
        for agent in self._agents.values():
            agent.reset()
        self._swarm_memory.clear()
        logger.debug("Swarm reset")
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"Swarm(agents={list(self._agents.keys())})"


__all__ = ["Swarm", "SwarmConfig", "SwarmResult", "SwarmHooks"]
