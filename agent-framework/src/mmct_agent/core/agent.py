"""Agent implementation with LLM interaction, tools, and memory."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Awaitable, TYPE_CHECKING
from uuid import uuid4

from mmct_agent.core.types import (
    Message,
    Role,
    AgentResponse,
    ToolCall,
    ToolResult,
    TokenUsage,
    StreamChunk,
    HandoffRequest,
)
from mmct_agent.core.exceptions import (
    AgentFrameworkError,
    LLMError,
    ToolExecutionError,
    HandoffError,
)
from mmct_agent.tools.registry import ToolRegistry
from mmct_agent.tools.executor import ToolExecutor
from mmct_agent.tools.decorator import get_tool_definition
from mmct_agent.memory.base import BaseMemory
from mmct_agent.memory.strategies import AdaptiveMemory
from mmct_agent.observability.logging import set_trace_id, clear_trace_id, get_logger

if TYPE_CHECKING:
    from mmct_agent.llm.base import BaseLLMClient
    from mmct_agent.tools.base import ToolDefinition

logger = get_logger(__name__)


# Callback types
OnMessageCallback = Callable[[Message], Awaitable[None] | None]
OnToolCallCallback = Callable[[ToolCall], Awaitable[None] | None]
OnToolResultCallback = Callable[[ToolResult], Awaitable[None] | None]
OnStreamCallback = Callable[[StreamChunk], Awaitable[None] | None]
OnHandoffCallback = Callable[[HandoffRequest], Awaitable[None] | None]


@dataclass
class AgentHooks:
    """Hooks for observing agent behavior."""
    
    on_message: OnMessageCallback | None = None
    on_tool_call: OnToolCallCallback | None = None
    on_tool_result: OnToolResultCallback | None = None
    on_stream: OnStreamCallback | None = None
    on_handoff: OnHandoffCallback | None = None


def _default_on_message(message: Message) -> None:
    """Default debug logging for messages."""
    role = message.role.value if hasattr(message.role, 'value') else str(message.role)
    content = message.content or ""
    tool_calls_count = len(message.tool_calls) if message.tool_calls else 0
    
    if role == "user":
        logger.debug(f"📥 USER:\n{content}")
    elif role == "assistant":
        if tool_calls_count > 0:
            if content:
                logger.debug(f"🤖 ASSISTANT: {content}\n   [+ {tool_calls_count} tool call(s)]")
            else:
                logger.debug(f"🤖 ASSISTANT: [requesting {tool_calls_count} tool call(s)]")
        else:
            logger.debug(f"🤖 ASSISTANT:\n{content}")
    elif role == "tool":
        logger.debug(f"🔧 TOOL RESPONSE:\n{content}")
    else:
        logger.debug(f"💬 {role.upper()}:\n{content}")


def _default_on_tool_call(tool_call: ToolCall) -> None:
    """Default debug logging for tool calls."""
    import json
    args_str = json.dumps(tool_call.arguments, indent=2) if tool_call.arguments else "{}"
    logger.debug(f"  ↳ 🔧 Calling tool: {tool_call.name}\n     Args: {args_str}")


def _default_on_tool_result(tool_result: ToolResult) -> None:
    """Default debug logging for tool results."""
    status = "✓" if tool_result.is_success else "✗"
    time_ms = tool_result.execution_time_ms or 0
    
    if tool_result.is_success:
        result_str = str(tool_result.result) if tool_result.result else ""
        logger.debug(f"  ↳ {status} Tool result: {tool_result.name} ({time_ms:.1f}ms)\n     Result: {result_str}")
    else:
        logger.debug(f"  ↳ {status} Tool failed: {tool_result.name}\n     Error: {tool_result.error}")


def _default_on_stream(chunk: StreamChunk) -> None:
    """Default debug logging for stream chunks."""
    if chunk.is_complete:
        tokens = chunk.token_usage.total_tokens if chunk.token_usage else 0
        logger.debug(f"📡 Stream completed ({tokens} tokens)")


def _default_on_handoff(handoff: HandoffRequest) -> None:
    """Default debug logging for handoff requests."""
    import json
    content_preview = json.dumps(handoff.context, indent=2) if handoff.context else "{}"
    logger.debug(f"  ↳ 🔀 Handoff → {handoff.target_agent}\n     Content: {content_preview}")


def get_default_agent_hooks() -> AgentHooks:
    """Get default agent hooks with debug logging."""
    return AgentHooks(
        on_message=_default_on_message,
        on_tool_call=_default_on_tool_call,
        on_tool_result=_default_on_tool_result,
        on_stream=_default_on_stream,
        on_handoff=_default_on_handoff,
    )


@dataclass
class _HandoffInfo:
    """Internal storage for handoff configuration."""
    
    target_agent: str
    description: str
    content_schema: dict[str, Any]  # JSON schema for handoff content


class Agent:
    """LLM-based agent with tool execution and memory management.
    
    An agent wraps an LLM client and provides:
    - Conversation memory management
    - Tool registration and execution
    - Streaming responses
    - Handoff capabilities
    - Observability hooks
    """
    
    # Special tool for handoff
    HANDOFF_TOOL_PREFIX = "handoff_to_"
    DEFAULT_TERMINATION_SEQUENCE = "TERMINATE"
    
    def __init__(
        self,
        name: str,
        llm_client: BaseLLMClient,
        system_prompt: str = "",
        tools: list[ToolDefinition | Callable[..., Any]] | None = None,
        memory: BaseMemory | None = None,
        hooks: AgentHooks | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        max_tool_iterations: int = 10,
        parallel_tool_calls: bool = True,
        tool_timeout_seconds: float = 30.0,
        stream_responses: bool = False,
        termination_sequence: str | None = "TERMINATE",
    ) -> None:
        """Initialize an agent.
        
        Args:
            name: Unique name for the agent.
            llm_client: LLM client for completions.
            system_prompt: System prompt defining agent behavior.
            tools: List of tools available to the agent.
            memory: Memory strategy for conversation history.
            hooks: Callback hooks for observability.
            metadata: Additional metadata about the agent.
            max_tool_iterations: Maximum number of tool execution iterations.
            parallel_tool_calls: Whether to execute tool calls in parallel.
            tool_timeout_seconds: Timeout for individual tool execution.
            stream_responses: Whether to stream LLM responses.
            termination_sequence: Sequence the LLM should include to signal completion.
                Set to None to disable (agent stops when no tool calls are made).
                Defaults to "TERMINATE".
        """
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.hooks = hooks or get_default_agent_hooks()
        self.metadata = metadata or {}
        
        # Configuration options
        self.max_tool_iterations = max_tool_iterations
        self.parallel_tool_calls = parallel_tool_calls
        self.tool_timeout_seconds = tool_timeout_seconds
        self.stream_responses = stream_responses
        self.termination_sequence = termination_sequence
        
        # Initialize memory
        self._memory = memory or AdaptiveMemory(llm_client)
        
        # Initialize tool registry and executor
        self._tool_registry = ToolRegistry()
        self._tool_executor = ToolExecutor(
            self._tool_registry,
            default_timeout=self.tool_timeout_seconds,
        )
        
        # Register provided tools
        if tools:
            for tool in tools:
                self._tool_registry.register(tool)
        
        # Handoff configurations
        self._handoffs: dict[str, _HandoffInfo] = {}
        
        # Tracking
        self._total_token_usage = TokenUsage()
        self._call_count = 0
    
    @property
    def memory(self) -> BaseMemory:
        """Get the agent's memory."""
        return self._memory
    
    @property
    def tools(self) -> list[ToolDefinition]:
        """Get list of available tools."""
        return self._tool_registry.list_tools()
    
    @property
    def total_token_usage(self) -> TokenUsage:
        """Get total token usage across all calls."""
        return self._total_token_usage
    
    def register_tool(self, tool: ToolDefinition | Callable[..., Any]) -> None:
        """Register a tool with the agent.
        
        Args:
            tool: Tool definition or decorated function.
        """
        self._tool_registry.register(tool)
        logger.debug(f"Registered tool with agent {self.name}")
    
    def register_handoff(
        self,
        target: str,
        *,
        description: str = "",
        content_schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a handoff to another agent.
        
        This creates a special tool that the agent can call to hand off
        to another agent. The content_schema defines the structured data
        the agent should provide when handing off.
        
        Args:
            target: Name of the target agent.
            description: Description of when to use this handoff.
            content_schema: JSON schema defining the handoff content structure.
                The schema should have "properties" with descriptions for each field.
                Example:
                    {
                        "type": "object",
                        "properties": {
                            "draft": {
                                "type": "string",
                                "description": "The complete article draft"
                            },
                            "notes": {
                                "type": "string", 
                                "description": "Any notes for the next agent"
                            }
                        },
                        "required": ["draft"]
                    }
        """
        tool_description = description or f"Hand off the conversation to {target} agent."
        
        # Default schema if none provided
        if content_schema is None:
            content_schema = {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to pass to the next agent.",
                    }
                },
                "required": ["content"],
            }
        
        handoff_info = _HandoffInfo(
            target_agent=target,
            description=tool_description,
            content_schema=content_schema,
        )
        
        self._handoffs[target] = handoff_info
        
        # Create handoff tool from schema
        tool_name = f"{self.HANDOFF_TOOL_PREFIX}{target}"
        
        handoff_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": content_schema,
            },
        }
        
        # Register as raw OpenAI format tool
        from mmct_agent.tools.base import ToolDefinition
        
        # Convert schema properties to ToolParameters
        parameters = []
        schema_props = content_schema.get("properties", {})
        required_fields = content_schema.get("required", [])
        
        from mmct_agent.tools.base import ToolParameter, ParameterType
        
        type_mapping = {
            "string": ParameterType.STRING,
            "number": ParameterType.NUMBER,
            "integer": ParameterType.INTEGER,
            "boolean": ParameterType.BOOLEAN,
            "array": ParameterType.ARRAY,
            "object": ParameterType.OBJECT,
        }
        
        for prop_name, prop_def in schema_props.items():
            prop_type = prop_def.get("type", "string")
            parameters.append(
                ToolParameter(
                    name=prop_name,
                    type=type_mapping.get(prop_type, ParameterType.STRING),
                    description=prop_def.get("description", ""),
                    required=prop_name in required_fields,
                    enum=prop_def.get("enum"),
                    items=prop_def.get("items"),
                    properties=prop_def.get("properties"),
                )
            )
        
        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=None,  # Handled specially
            is_async=True,
        )
        
        self._tool_registry.register(tool_def)
        logger.debug(f"Registered handoff from {self.name} to {target}")
    
    async def run(
        self,
        user_input: str | Message,
        trace_id: str | None = None,
    ) -> AgentResponse:
        """Run the agent with user input.
        
        Args:
            user_input: User message or string.
            trace_id: Optional trace ID for observability.
            
        Returns:
            AgentResponse with the final result.
        """
        # Determine if we own the trace_id lifecycle (standalone run vs swarm)
        owns_trace_id = trace_id is None
        trace_id = trace_id or str(uuid4())
        
        if owns_trace_id:
            set_trace_id(trace_id)  # Set trace_id for all subsequent logs
        
        start_time = time.perf_counter()
        self._call_count += 1
        
        logger.debug(f"Agent '{self.name}' run #{self._call_count} starting")
        
        try:
            # Convert string to message
            if isinstance(user_input, str):
                user_input = Message.user(user_input)
            
            # Add to memory
            await self._memory.add(user_input)
            await self._trigger_hook(self.hooks.on_message, user_input)
            
            # Collect all messages for response
            all_messages: list[Message] = [user_input]
            all_tool_results: list[ToolResult] = []
            
            # Run agent loop (tool execution)
            iteration = 0
            final_content: str | None = None
            handoff_request: HandoffRequest | None = None
            
            while iteration < self.max_tool_iterations:
                iteration += 1
                
                # Get context for LLM
                system_prompt_with_termination = self._build_system_prompt()
                system_message = Message.system(system_prompt_with_termination) if system_prompt_with_termination else None
                context_messages = await self._memory.get_context_for_llm(system_message)
                
                # Get tool definitions
                tool_defs = self._tool_registry.list_tools() if self._tool_registry else None
                
                # Call LLM
                if self.stream_responses:
                    response = await self._run_streaming(context_messages, tool_defs, trace_id)
                else:
                    response = await self.llm_client.complete(context_messages, tool_defs)
                
                # Update token usage
                self._total_token_usage = self._total_token_usage + response.token_usage
                
                # Create assistant message
                assistant_message = Message.assistant(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                await self._memory.add(assistant_message)
                all_messages.append(assistant_message)
                await self._trigger_hook(self.hooks.on_message, assistant_message)
                
                # Check for tool calls
                if not response.tool_calls:
                    # No tool calls - check for termination sequence
                    if self._should_terminate(response.content):
                        final_content = self._strip_termination_sequence(response.content)
                        break
                    # No termination sequence and no tool calls - continue iterating
                    # (LLM should include termination sequence when done)
                    logger.debug(
                        f"No tool calls and no termination sequence '{self.termination_sequence}' found. "
                        f"Continuing iteration {iteration}/{self.max_tool_iterations}..."
                    )
                    # Add a user message prompting for completion or action
                    prompt_msg = Message.user(
                        f"Please either use a tool to continue, hand off to another agent, "
                        f"or provide your COMPLETE final answer ending with '{self.termination_sequence}'."
                    )
                    await self._memory.add(prompt_msg)
                    all_messages.append(prompt_msg)
                    continue
                
                # Check for handoff
                handoff_request = self._check_for_handoff(response.tool_calls)
                if handoff_request:
                    await self._trigger_hook(self.hooks.on_handoff, handoff_request)
                    break
                
                # Execute tools
                for tc in response.tool_calls:
                    await self._trigger_hook(self.hooks.on_tool_call, tc)
                
                if self.parallel_tool_calls:
                    tool_results = await self._tool_executor.execute_parallel(response.tool_calls)
                else:
                    tool_results = []
                    for tc in response.tool_calls:
                        result = await self._tool_executor.execute(tc)
                        tool_results.append(result)
                
                # Add tool results to memory and track
                for result in tool_results:
                    await self._trigger_hook(self.hooks.on_tool_result, result)
                    tool_message = Message.tool_result(result)
                    await self._memory.add(tool_message)
                    all_messages.append(tool_message)
                    all_tool_results.append(result)
            
            # If we exhausted iterations without proper termination, use last content
            if final_content is None and not handoff_request:
                # Get the last assistant message content
                for msg in reversed(all_messages):
                    if msg.role == Role.ASSISTANT and msg.content:
                        final_content = self._strip_termination_sequence(msg.content)
                        logger.warning(
                            f"Agent '{self.name}' reached max iterations without termination sequence. "
                            f"Using last response."
                        )
                        break
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            logger.debug(
                f"Agent '{self.name}' run #{self._call_count} completed: "
                f"{iteration} iterations, {len(all_tool_results)} tool calls, {latency_ms:.0f}ms"
            )
            
            return AgentResponse(
                content=final_content,
                messages=all_messages,
                tool_results=all_tool_results,
                token_usage=self._total_token_usage,
                latency_ms=latency_ms,
                agent_name=self.name,
                handoff_to=handoff_request.target_agent if handoff_request else None,
                handoff_context=handoff_request.context if handoff_request else None,
                trace_id=trace_id,
            )
        finally:
            if owns_trace_id:
                clear_trace_id()  # Only clear if we set it
    
    async def run_stream(
        self,
        user_input: str | Message,
        trace_id: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run the agent with streaming response.
        
        Args:
            user_input: User message or string.
            trace_id: Optional trace ID.
            
        Yields:
            StreamChunk objects with partial content.
        """
        trace_id = trace_id or str(uuid4())
        
        # Convert string to message
        if isinstance(user_input, str):
            user_input = Message.user(user_input)
        
        # Add to memory
        await self._memory.add(user_input)
        
        # Get context for LLM
        system_message = Message.system(self.system_prompt) if self.system_prompt else None
        context_messages = await self._memory.get_context_for_llm(system_message)
        
        # Get tool definitions
        tool_defs = self._tool_registry.list_tools() if self._tool_registry else None
        
        # Stream response
        full_content = ""
        async for chunk in self.llm_client.complete_stream(context_messages, tool_defs):
            full_content += chunk.content
            await self._trigger_hook(self.hooks.on_stream, chunk)
            yield chunk
            
            if chunk.is_complete:
                # Update token usage
                if chunk.token_usage:
                    self._total_token_usage = self._total_token_usage + chunk.token_usage
                
                # Add to memory
                assistant_message = Message.assistant(
                    content=full_content if full_content else None,
                    tool_calls=chunk.tool_calls,
                )
                await self._memory.add(assistant_message)
    
    async def _run_streaming(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        trace_id: str,
    ) -> Any:
        """Run with streaming and collect full response.
        
        Args:
            messages: Context messages.
            tools: Tool definitions.
            trace_id: Trace ID.
            
        Returns:
            LLMResponse equivalent.
        """
        from mmct_agent.llm.base import LLMResponse
        
        full_content = ""
        final_tool_calls = None
        final_usage = TokenUsage()
        
        async for chunk in self.llm_client.complete_stream(messages, tools):
            full_content += chunk.content
            await self._trigger_hook(self.hooks.on_stream, chunk)
            
            if chunk.is_complete:
                final_tool_calls = chunk.tool_calls
                if chunk.token_usage:
                    final_usage = chunk.token_usage
        
        return LLMResponse(
            content=full_content if full_content else None,
            tool_calls=final_tool_calls,
            token_usage=final_usage,
            finish_reason="stop",
            model="",
            latency_ms=0,
        )
    
    def _should_terminate(self, content: str | None) -> bool:
        """Check if the content contains the termination sequence.
        
        Args:
            content: LLM response content.
            
        Returns:
            True if termination sequence is found or not required.
        """
        # If no termination sequence is set, any response without tool calls terminates
        if self.termination_sequence is None:
            return True
        
        if not content:
            return False
        
        return self.termination_sequence in content
    
    def _build_system_prompt(self) -> str | None:
        """Build the system prompt including handoffs and termination sequence instruction.
        
        Returns:
            System prompt with handoffs and termination instruction appended, or None if no prompt.
        """
        if not self.system_prompt and not self.termination_sequence and not self._handoffs:
            return None
        
        parts: list[str] = []
        
        if self.system_prompt:
            parts.append(self.system_prompt)
        
        # Add handoff descriptions if any are registered
        if self._handoffs:
            handoff_lines = ["\n\nAVAILABLE HANDOFFS:"]
            for target, info in self._handoffs.items():
                handoff_lines.append(f"- handoff_to_{target}: {info.description}")
            handoff_lines.append(
                "\nUse the appropriate handoff tool when the described conditions are met."
            )
            parts.append("\n".join(handoff_lines))
        
        if self.termination_sequence:
            termination_instruction = (
                f"\n\nIMPORTANT: When you have completed your task and are ready to provide your final answer, "
                f"you MUST include '{self.termination_sequence}' at the end of your response. "
                f"Your final response should contain your COMPLETE answer followed by '{self.termination_sequence}'. "
                f"Do not use '{self.termination_sequence}' until you have fully completed the task."
            )
            parts.append(termination_instruction)
        
        return "".join(parts) if parts else None
    
    def _strip_termination_sequence(self, content: str | None) -> str | None:
        """Remove the termination sequence from content.
        
        Args:
            content: LLM response content.
            
        Returns:
            Content with termination sequence removed.
        """
        if not content or not self.termination_sequence:
            return content
        
        # Remove the termination sequence and clean up whitespace
        cleaned = content.replace(self.termination_sequence, "").strip()
        return cleaned if cleaned else None
    
    def _check_for_handoff(self, tool_calls: list[ToolCall]) -> HandoffRequest | None:
        """Check if any tool call is a handoff request.
        
        Args:
            tool_calls: List of tool calls from LLM.
            
        Returns:
            HandoffRequest if found, None otherwise.
        """
        for tc in tool_calls:
            if tc.name.startswith(self.HANDOFF_TOOL_PREFIX):
                target = tc.name[len(self.HANDOFF_TOOL_PREFIX):]
                if target in self._handoffs:
                    return HandoffRequest(
                        target_agent=target,
                        context=tc.arguments,  # Pass all arguments as context
                    )
        return None
    
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
            logger.warning(f"Hook raised exception: {e}", exc_info=True)
    
    async def get_context_summary(self, max_tokens: int = 500) -> str:
        """Get a summary of the agent's current context.
        
        Useful for handoff context transformation.
        
        Args:
            max_tokens: Maximum tokens for summary.
            
        Returns:
            Summary string.
        """
        messages = await self._memory.get_messages()
        
        # Build conversation text
        lines: list[str] = []
        for msg in messages[-10:]:  # Last 10 messages
            role = msg.role.value.upper()
            content = msg.content or ""
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"{role}: {content}")
        
        conversation = "\n".join(lines)
        
        # Use LLM to summarize if content is long
        from mmct_agent.memory.base import InMemoryTokenCounter
        counter = InMemoryTokenCounter()
        
        if counter.count(conversation) > max_tokens:
            summary_prompt = f"""Summarize this conversation concisely in under {max_tokens} tokens:

{conversation}

Summary:"""
            response = await self.llm_client.complete([Message.user(summary_prompt)])
            return response.content or conversation
        
        return conversation
    
    def reset(self) -> None:
        """Reset the agent's memory and state."""
        self._memory.clear()
        self._total_token_usage = TokenUsage()
        self._call_count = 0
        logger.debug(f"Agent {self.name} reset")
    
    async def save_memory(
        self,
        path: str = "./memory_logs",
        session_id: str | None = None,
    ) -> str:
        """Save agent memory to disk for debugging.
        
        Args:
            path: Base directory for memory logs.
            session_id: Optional session identifier. Auto-generated if not provided.
            
        Returns:
            Path where memory was saved.
        """
        from mmct_agent.memory.persistence import MemoryPersistence
        
        persistence = MemoryPersistence(base_path=path, session_id=session_id)
        await persistence.save_agent_memory(self.name, self._memory)
        
        logger.info(f"Agent {self.name} memory saved to {path}")
        return str(persistence._session_path)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Agent(name={self.name!r}, "
            f"tools={len(self._tool_registry)}, "
            f"memory={self._memory.strategy_name})")
