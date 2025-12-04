"""Tool registry for managing available tools."""

from __future__ import annotations

from typing import Any, Callable

from mmct_agent.core.exceptions import ToolNotFoundError
from mmct_agent.tools.base import ToolDefinition
from mmct_agent.tools.decorator import get_tool_definition
from mmct_agent.observability.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing tools available to agents.
    
    The registry stores tool definitions and provides lookup by name.
    Tools can be registered either as ToolDefinition objects or as
    decorated functions.
    """
    
    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(self, tool: ToolDefinition | Callable[..., Any]) -> None:
        """Register a tool in the registry.
        
        Args:
            tool: Either a ToolDefinition or a @tool decorated function.
            
        Raises:
            ValueError: If tool is not a valid tool definition.
        """
        if isinstance(tool, ToolDefinition):
            tool_def = tool
        else:
            # Try to get tool definition from decorated function
            tool_def = get_tool_definition(tool)
            if tool_def is None:
                raise ValueError(
                    f"Function {getattr(tool, '__name__', tool)} is not decorated with @tool. "
                    "Use the @tool decorator or pass a ToolDefinition."
                )
        
        if tool_def.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool_def.name}")
        
        self._tools[tool_def.name] = tool_def
        logger.debug(f"Registered tool: {tool_def.name}")
    
    def register_many(self, tools: list[ToolDefinition | Callable[..., Any]]) -> None:
        """Register multiple tools at once.
        
        Args:
            tools: List of tools to register.
        """
        for tool in tools:
            self.register(tool)
    
    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove.
            
        Returns:
            True if tool was removed, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False
    
    def get(self, name: str) -> ToolDefinition:
        """Get a tool by name.
        
        Args:
            name: Tool name.
            
        Returns:
            ToolDefinition for the tool.
            
        Raises:
            ToolNotFoundError: If tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(
                message=f"Tool '{name}' not found in registry",
                tool_name=name,
                available_tools=list(self._tools.keys()),
            )
        return self._tools[name]
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name.
            
        Returns:
            True if tool is registered.
        """
        return name in self._tools
    
    def list_tools(self) -> list[ToolDefinition]:
        """Get all registered tools.
        
        Returns:
            List of all tool definitions.
        """
        return list(self._tools.values())
    
    def list_names(self) -> list[str]:
        """Get names of all registered tools.
        
        Returns:
            List of tool names.
        """
        return list(self._tools.keys())
    
    def clear(self) -> None:
        """Remove all tools from the registry."""
        self._tools.clear()
        logger.debug("Cleared tool registry")
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool name is in registry."""
        return name in self._tools
    
    def __iter__(self):
        """Iterate over tool definitions."""
        return iter(self._tools.values())
