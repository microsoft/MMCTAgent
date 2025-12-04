"""Memory persistence for debugging and recovery."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import aiofiles

from mmct_agent.observability.logging import get_logger

if TYPE_CHECKING:
    from mmct_agent.core.types import Message
    from mmct_agent.memory.base import BaseMemory
    from mmct_agent.memory.swarm_memory import SwarmMemory

logger = get_logger(__name__)


class MemoryPersistence:
    """Handles persistence of memory to disk for debugging.
    
    Saves:
    - All messages with timestamps
    - Memory operations log
    - Swarm state
    """
    
    def __init__(
        self,
        base_path: str | Path = "./memory_logs",
        session_id: str | None = None,
    ) -> None:
        """Initialize persistence handler.
        
        Args:
            base_path: Base directory for memory logs.
            session_id: Optional session identifier.
        """
        self._base_path = Path(base_path)
        self._session_id = session_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._session_path = self._base_path / self._session_id
        self._initialized = False
    
    async def initialize(self) -> None:
        """Create session directory structure."""
        if self._initialized:
            return
        
        self._session_path.mkdir(parents=True, exist_ok=True)
        (self._session_path / "agents").mkdir(exist_ok=True)
        (self._session_path / "swarm").mkdir(exist_ok=True)
        
        # Create session metadata
        metadata = {
            "session_id": self._session_id,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0",
        }
        await self._write_json(self._session_path / "metadata.json", metadata)
        
        self._initialized = True
        logger.info(f"Memory persistence initialized at {self._session_path}")
    
    async def save_agent_memory(
        self,
        agent_name: str,
        memory: BaseMemory,
    ) -> None:
        """Save agent memory to disk.
        
        Args:
            agent_name: Name of the agent.
            memory: Agent's memory instance.
        """
        await self.initialize()
        
        agent_path = self._session_path / "agents" / agent_name
        agent_path.mkdir(exist_ok=True)
        
        # Save messages
        messages = memory.get_raw_messages()
        messages_data = [self._message_to_dict(m) for m in messages]
        await self._write_json(agent_path / "messages.json", messages_data)
        
        # Save operation log
        operations = memory.get_operation_log()
        await self._write_json(agent_path / "operations.json", operations)
        
        # Save metadata
        metadata = {
            "agent_name": agent_name,
            "strategy": memory.strategy_name,
            "message_count": len(messages),
            "saved_at": datetime.utcnow().isoformat(),
        }
        await self._write_json(agent_path / "metadata.json", metadata)
        
        logger.debug(f"Saved memory for agent {agent_name}")
    
    async def save_swarm_memory(self, swarm_memory: SwarmMemory) -> None:
        """Save swarm memory to disk.
        
        Args:
            swarm_memory: Swarm memory instance.
        """
        await self.initialize()
        
        swarm_path = self._session_path / "swarm"
        
        # Save state
        state = swarm_memory.state
        state_data = {
            "current_agent": state.current_agent,
            "visited_agents": state.visited_agents,
            "handoff_count": state.handoff_count,
            "total_iterations": state.total_iterations,
            "shared_data": state.shared_data,
            "agent_contexts": {
                name: {
                    "agent_name": ctx.agent_name,
                    "summary": ctx.summary,
                    "key_findings": ctx.key_findings,
                    "metadata": ctx.metadata,
                    "timestamp": ctx.timestamp.isoformat(),
                    "token_usage": ctx.token_usage,
                }
                for name, ctx in state.agent_contexts.items()
            },
        }
        await self._write_json(swarm_path / "state.json", state_data)
        
        # Save operation log
        operations = swarm_memory.get_operation_log()
        await self._write_json(swarm_path / "operations.json", operations)
        
        logger.debug("Saved swarm memory")
    
    async def save_all(
        self,
        agent_memories: dict[str, BaseMemory],
        swarm_memory: SwarmMemory | None = None,
    ) -> None:
        """Save all memories to disk.
        
        Args:
            agent_memories: Dict mapping agent names to their memories.
            swarm_memory: Optional swarm memory.
        """
        for agent_name, memory in agent_memories.items():
            await self.save_agent_memory(agent_name, memory)
        
        if swarm_memory:
            await self.save_swarm_memory(swarm_memory)
    
    async def load_agent_messages(self, agent_name: str) -> list[dict[str, Any]]:
        """Load saved messages for an agent.
        
        Args:
            agent_name: Name of the agent.
            
        Returns:
            List of message dictionaries.
        """
        messages_path = self._session_path / "agents" / agent_name / "messages.json"
        if not messages_path.exists():
            return []
        
        return await self._read_json(messages_path)
    
    async def load_swarm_state(self) -> dict[str, Any] | None:
        """Load saved swarm state.
        
        Returns:
            Swarm state dictionary or None.
        """
        state_path = self._session_path / "swarm" / "state.json"
        if not state_path.exists():
            return None
        
        return await self._read_json(state_path)
    
    async def append_operation(
        self,
        category: str,
        name: str,
        operation: dict[str, Any],
    ) -> None:
        """Append an operation to the log incrementally.
        
        Args:
            category: "agents" or "swarm".
            name: Agent name or empty for swarm.
            operation: Operation to log.
        """
        await self.initialize()
        
        if category == "agents":
            log_path = self._session_path / "agents" / name / "operations_incremental.jsonl"
        else:
            log_path = self._session_path / "swarm" / "operations_incremental.jsonl"
        
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(log_path, "a") as f:
            await f.write(json.dumps(operation, default=str) + "\n")
    
    def _message_to_dict(self, message: Message) -> dict[str, Any]:
        """Convert a message to a serializable dictionary."""
        data: dict[str, Any] = {
            "id": message.id,
            "role": message.role.value,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata,
        }
        
        if message.name:
            data["name"] = message.name
        
        if message.tool_call_id:
            data["tool_call_id"] = message.tool_call_id
        
        if message.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in message.tool_calls
            ]
        
        return data
    
    async def _write_json(self, path: Path, data: Any) -> None:
        """Write data to JSON file."""
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2, default=str))
    
    async def _read_json(self, path: Path) -> Any:
        """Read data from JSON file."""
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
            return json.loads(content)
