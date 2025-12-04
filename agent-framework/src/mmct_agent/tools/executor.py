"""Tool executor for running tools with parallel execution and timeout handling."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from mmct_agent.core.exceptions import ToolExecutionError, ToolTimeoutError, ToolNotFoundError
from mmct_agent.core.types import ToolCall, ToolResult
from mmct_agent.tools.base import ToolDefinition
from mmct_agent.tools.registry import ToolRegistry
from mmct_agent.observability.logging import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """Executor for running tools with parallel execution support.
    
    Handles both sync and async tools, with timeout handling and
    comprehensive error reporting.
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float = 30.0,
        max_parallel: int = 10,
    ) -> None:
        """Initialize the tool executor.
        
        Args:
            registry: Tool registry to look up tools.
            default_timeout: Default timeout in seconds for tool execution.
            max_parallel: Maximum number of tools to execute in parallel.
        """
        self._registry = registry
        self._default_timeout = default_timeout
        self._max_parallel = max_parallel
    
    async def execute(
        self,
        tool_call: ToolCall,
        timeout: float | None = None,
    ) -> ToolResult:
        """Execute a single tool call.
        
        Args:
            tool_call: Tool call to execute.
            timeout: Timeout in seconds. Uses tool's timeout or default if None.
            
        Returns:
            ToolResult with execution result or error.
        """
        start_time = time.perf_counter()
        
        try:
            # Get tool definition
            tool_def = self._registry.get(tool_call.name)
        except ToolNotFoundError as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=f"Tool not found: {tool_call.name}",
                execution_time_ms=0.0,
            )
        
        # Determine timeout
        effective_timeout = timeout or tool_def.timeout_seconds or self._default_timeout
        
        try:
            # Execute the tool
            result = await self._execute_tool(
                tool_def,
                tool_call.arguments,
                effective_timeout,
            )
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result,
                error=None,
                execution_time_ms=execution_time_ms,
            )
            
        except asyncio.TimeoutError:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Tool execution timed out after {effective_timeout}s"
            
            logger.warning(
                error_msg,
                extra={
                    "tool_name": tool_call.name,
                    "timeout_seconds": effective_timeout,
                },
            )
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )
            
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Tool execution failed: {type(e).__name__}: {str(e)}"
            
            logger.error(
                error_msg,
                extra={
                    "tool_name": tool_call.name,
                    "error_type": type(e).__name__,
                    "arguments": tool_call.arguments,
                },
                exc_info=True,
            )
            
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )
    
    async def execute_parallel(
        self,
        tool_calls: list[ToolCall],
        timeout: float | None = None,
    ) -> list[ToolResult]:
        """Execute multiple tool calls in parallel.
        
        Args:
            tool_calls: List of tool calls to execute.
            timeout: Timeout per tool. Uses tool's timeout or default if None.
            
        Returns:
            List of ToolResults in the same order as input.
        """
        if not tool_calls:
            return []
        
        # Limit parallelism
        semaphore = asyncio.Semaphore(self._max_parallel)
        
        async def execute_with_semaphore(tool_call: ToolCall) -> ToolResult:
            async with semaphore:
                return await self.execute(tool_call, timeout)
        
        # Execute all tools in parallel
        tasks = [execute_with_semaphore(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results
    
    async def _execute_tool(
        self,
        tool_def: ToolDefinition,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        """Execute a tool with timeout.
        
        Args:
            tool_def: Tool definition with callable.
            arguments: Arguments to pass to the tool.
            timeout: Timeout in seconds.
            
        Returns:
            Tool execution result.
            
        Raises:
            asyncio.TimeoutError: If execution times out.
            Exception: Any exception raised by the tool.
        """
        if tool_def.func is None:
            raise ToolExecutionError(
                message=f"Tool {tool_def.name} has no callable function",
                tool_name=tool_def.name,
                arguments=arguments,
            )
        
        # Validate and filter arguments
        valid_args = self._validate_arguments(tool_def, arguments)
        
        if tool_def.is_async:
            # Async tool - call directly with timeout
            return await asyncio.wait_for(
                tool_def.func(**valid_args),
                timeout=timeout,
            )
        else:
            # Sync tool - run in executor with timeout
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: tool_def.func(**valid_args)),
                timeout=timeout,
            )
    
    def _validate_arguments(
        self,
        tool_def: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and filter arguments for a tool.
        
        Args:
            tool_def: Tool definition.
            arguments: Provided arguments.
            
        Returns:
            Validated and filtered arguments.
            
        Raises:
            ToolExecutionError: If required arguments are missing.
        """
        valid_param_names = {p.name for p in tool_def.parameters}
        required_params = {p.name for p in tool_def.parameters if p.required}
        
        # Check for missing required parameters
        provided_params = set(arguments.keys())
        missing = required_params - provided_params
        
        if missing:
            raise ToolExecutionError(
                message=f"Missing required parameters for tool {tool_def.name}: {missing}",
                tool_name=tool_def.name,
                arguments=arguments,
            )
        
        # Filter to only valid parameters (ignore extra ones from LLM)
        filtered = {k: v for k, v in arguments.items() if k in valid_param_names}
        
        # Add defaults for missing optional parameters
        for param in tool_def.parameters:
            if param.name not in filtered and param.default is not None:
                filtered[param.name] = param.default
        
        return filtered
