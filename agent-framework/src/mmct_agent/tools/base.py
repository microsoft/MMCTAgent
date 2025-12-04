"""Base tool definitions and types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from enum import Enum


class ParameterType(str, Enum):
    """JSON Schema parameter types."""
    
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None  # For array types
    properties: dict[str, Any] | None = None  # For object types
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }
        
        if self.enum:
            schema["enum"] = self.enum
            
        if self.items and self.type == ParameterType.ARRAY:
            schema["items"] = self.items
            
        if self.properties and self.type == ParameterType.OBJECT:
            schema["properties"] = self.properties
            
        if self.default is not None:
            schema["default"] = self.default
            
        return schema


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by an LLM."""
    
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    func: Callable[..., Any] | Callable[..., Awaitable[Any]] | None = None
    is_async: bool = False
    timeout_seconds: float = 30.0
    
    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        # Build properties and required lists
        properties: dict[str, Any] = {}
        required: list[str] = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    
    def __hash__(self) -> int:
        """Make ToolDefinition hashable by name."""
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        """Check equality by name."""
        if isinstance(other, ToolDefinition):
            return self.name == other.name
        return False
