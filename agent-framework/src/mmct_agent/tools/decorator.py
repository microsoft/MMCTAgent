"""Tool decorator for easy tool creation from functions."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, TypeVar, get_type_hints, get_origin, get_args, Union

from mmct_agent.tools.base import ToolDefinition, ToolParameter, ParameterType
from mmct_agent.observability.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _python_type_to_parameter_type(py_type: type) -> ParameterType:
    """Convert Python type to JSON Schema parameter type.
    
    Args:
        py_type: Python type annotation.
        
    Returns:
        Corresponding ParameterType.
    """
    # Handle Optional types (Union[X, None])
    origin = get_origin(py_type)
    if origin is Union:
        args = get_args(py_type)
        # Filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            py_type = non_none_args[0]
    
    # Handle list/List types
    if origin is list or py_type is list:
        return ParameterType.ARRAY
    
    # Handle dict/Dict types
    if origin is dict or py_type is dict:
        return ParameterType.OBJECT
    
    # Map basic types
    type_mapping: dict[type, ParameterType] = {
        str: ParameterType.STRING,
        int: ParameterType.INTEGER,
        float: ParameterType.NUMBER,
        bool: ParameterType.BOOLEAN,
        list: ParameterType.ARRAY,
        dict: ParameterType.OBJECT,
    }
    
    return type_mapping.get(py_type, ParameterType.STRING)


def _extract_parameters_from_function(func: Callable[..., Any]) -> list[ToolParameter]:
    """Extract parameter definitions from function signature.
    
    Args:
        func: Function to extract parameters from.
        
    Returns:
        List of ToolParameter definitions.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    
    # Parse docstring for parameter descriptions
    docstring = inspect.getdoc(func) or ""
    param_docs = _parse_docstring_params(docstring)
    
    parameters: list[ToolParameter] = []
    
    for name, param in sig.parameters.items():
        # Skip self, cls, *args, **kwargs
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        
        # Get type annotation
        py_type = hints.get(name, str)
        param_type = _python_type_to_parameter_type(py_type)
        
        # Check if parameter is optional (has default value or is Optional type)
        is_required = param.default is inspect.Parameter.empty
        default_value = None if param.default is inspect.Parameter.empty else param.default
        
        # Check if type is Optional
        origin = get_origin(py_type)
        if origin is Union:
            args = get_args(py_type)
            if type(None) in args:
                is_required = False
        
        # Get description from docstring
        description = param_docs.get(name, f"Parameter: {name}")
        
        # Handle array item types
        items = None
        if param_type == ParameterType.ARRAY:
            origin = get_origin(py_type)
            if origin is list:
                args = get_args(py_type)
                if args:
                    item_type = _python_type_to_parameter_type(args[0])
                    items = {"type": item_type.value}
        
        parameters.append(
            ToolParameter(
                name=name,
                type=param_type,
                description=description,
                required=is_required,
                default=default_value,
                items=items,
            )
        )
    
    return parameters


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Parse parameter descriptions from docstring.
    
    Supports Google-style, NumPy-style, and Sphinx-style docstrings.
    
    Args:
        docstring: Function docstring.
        
    Returns:
        Dict mapping parameter names to descriptions.
    """
    params: dict[str, str] = {}
    
    if not docstring:
        return params
    
    lines = docstring.split("\n")
    current_param: str | None = None
    current_desc: list[str] = []
    in_params_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check for Args/Parameters section
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_params_section = True
            continue
        
        # Check for end of params section
        if in_params_section and stripped.endswith(":") and not stripped.startswith(":"):
            if not any(c in stripped.lower() for c in ("param", "(", ")")):
                # Likely a new section
                if current_param:
                    params[current_param] = " ".join(current_desc).strip()
                in_params_section = False
                continue
        
        # Parse Google-style: param_name: description or param_name (type): description
        if in_params_section:
            # Check for new parameter
            if ":" in stripped and not stripped.startswith(" "):
                # Save previous parameter
                if current_param:
                    params[current_param] = " ".join(current_desc).strip()
                
                # Parse new parameter
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()
                
                # Handle "param_name (type)" format
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()
                
                current_param = param_part
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif current_param and stripped:
                # Continuation of description
                current_desc.append(stripped)
        
        # Parse Sphinx-style: :param name: description
        if stripped.startswith(":param "):
            if current_param:
                params[current_param] = " ".join(current_desc).strip()
            
            rest = stripped[7:]  # Remove ":param "
            if ":" in rest:
                parts = rest.split(":", 1)
                current_param = parts[0].strip()
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
    
    # Save last parameter
    if current_param:
        params[current_param] = " ".join(current_desc).strip()
    
    return params


def tool(
    name: str | None = None,
    description: str | None = None,
    timeout_seconds: float = 30.0,
) -> Callable[[F], F]:
    """Decorator to convert a function into a tool.
    
    The decorator extracts parameter information from type hints and docstrings
    to create a complete tool definition.
    
    Args:
        name: Tool name. Defaults to function name.
        description: Tool description. Defaults to function docstring.
        timeout_seconds: Timeout for tool execution.
        
    Returns:
        Decorated function with _tool_definition attribute.
        
    Example:
        @tool(description="Add two numbers")
        async def add(a: int, b: int) -> int:
            '''Add two numbers together.
            
            Args:
                a: First number.
                b: Second number.
                
            Returns:
                Sum of a and b.
            '''
            return a + b
    """
    def decorator(func: F) -> F:
        # Get function metadata
        func_name = name or func.__name__
        
        # Get description from docstring if not provided
        func_description = description
        if not func_description:
            docstring = inspect.getdoc(func) or ""
            # Use first paragraph as description
            paragraphs = docstring.split("\n\n")
            func_description = paragraphs[0].replace("\n", " ").strip() if paragraphs else f"Tool: {func_name}"
        
        # Extract parameters
        parameters = _extract_parameters_from_function(func)
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        # Create tool definition
        tool_def = ToolDefinition(
            name=func_name,
            description=func_description,
            parameters=parameters,
            func=func,
            is_async=is_async,
            timeout_seconds=timeout_seconds,
        )
        
        # Attach definition to function
        func._tool_definition = tool_def  # type: ignore
        
        logger.debug(f"Registered tool: {func_name}", extra={"parameters": [p.name for p in parameters]})
        
        return func
    
    return decorator


def get_tool_definition(func: Callable[..., Any]) -> ToolDefinition | None:
    """Get the tool definition from a decorated function.
    
    Args:
        func: Potentially decorated function.
        
    Returns:
        ToolDefinition if function is decorated, None otherwise.
    """
    return getattr(func, "_tool_definition", None)
