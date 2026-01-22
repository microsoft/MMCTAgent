"""
Stream handling utilities for V2 orchestrator.
Converts AutoGen message streams to console output or dict streams for API consumption.
"""

import json
import time
from datetime import datetime
from autogen_agentchat.messages import TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent, HandoffMessage
from autogen_agentchat.base import TaskResult
from autogen_core.models import RequestUsage


async def console_stream_generator(stream_gen):
    """
    Helper to print stream events to console while yielding raw messages.
    Used for non-streaming mode with console output.
    """
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async for message in stream_gen:
        _print_message_to_console(message, start_time, total_usage)
        yield message


async def dict_stream_generator(stream_gen):
    """
    Convert stream events to dictionaries for API consumption.
    Yields structured dicts suitable for JSON serialization/SSE.
    """
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async for message in stream_gen:
        result = _message_to_dict(message, start_time, total_usage)
        if result:
            yield result


async def dict_stream_generator_with_console(stream_gen):
    """
    Convert stream events to dictionaries AND print to console.
    Useful for debugging while streaming to API.
    """
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
    
    async for message in stream_gen:
        # Print to console
        _print_message_to_console(message, start_time, total_usage)
        
        # Yield dict
        result = _message_to_dict(message, start_time, total_usage)
        if result:
            yield result


def _message_to_dict(message, start_time: float, total_usage: RequestUsage) -> dict | None:
    """
    Convert a single message to a dict for API consumption.
    Updates total_usage in place.
    """
    source = getattr(message, "source", "system")
    timestamp = datetime.now().isoformat()
    
    # Accumulate token usage
    if hasattr(message, "models_usage") and message.models_usage:
        total_usage.completion_tokens += message.models_usage.completion_tokens
        total_usage.prompt_tokens += message.models_usage.prompt_tokens
    
    if isinstance(message, TaskResult):
        duration = time.time() - start_time
        final_content = message.messages[-1].content if message.messages else ""
        
        # Try to parse JSON from final content
        parsed_content = final_content
        try:
            clean = final_content.replace("```json", "").replace("```", "").replace("TERMINATE", "").strip()
            parsed_content = json.loads(clean)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return {
            "type": "result",
            "source": source,
            "content": parsed_content,
            "message_count": len(message.messages),
            "stop_reason": message.stop_reason,
            "duration_seconds": round(duration, 2),
            "token_usage": {
                "prompt_tokens": total_usage.prompt_tokens,
                "completion_tokens": total_usage.completion_tokens
            },
            "timestamp": timestamp
        }
    
    elif isinstance(message, TextMessage):
        return {
            "type": "message",
            "source": source,
            "content": message.content,
            "timestamp": timestamp
        }
    
    elif isinstance(message, HandoffMessage):
        return {
            "type": "handoff",
            "source": source,
            "target": message.target,
            "content": message.content,
            "timestamp": timestamp
        }
    
    elif isinstance(message, ToolCallRequestEvent):
        tool_names = [tc.name for tc in message.content]
        return {
            "type": "tool_call",
            "source": source,
            "tool_names": tool_names,
            "tools": [{"name": tc.name, "arguments": tc.arguments} for tc in message.content],
            "timestamp": timestamp
        }
    
    elif isinstance(message, ToolCallExecutionEvent):
        return {
            "type": "tool_result",
            "source": source,
            "results": [
                {
                    "call_id": r.call_id,
                    "content": str(r.content)[:500] + "..." if len(str(r.content)) > 500 else str(r.content)
                }
                for r in message.content
            ],
            "timestamp": timestamp
        }
    
    return None


def _print_message_to_console(message, start_time: float, total_usage: RequestUsage):
    """Print a message to console with formatting."""
    source = getattr(message, "source", "Unknown")
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # Accumulate token usage
    if hasattr(message, "models_usage") and message.models_usage:
        total_usage.completion_tokens += message.models_usage.completion_tokens
        total_usage.prompt_tokens += message.models_usage.prompt_tokens
    
    if isinstance(message, TaskResult):
        duration = time.time() - start_time
        print(f"\n\033[90m[{now}]\033[0m \033[95m[System]\033[0m: Task Completed.")
        print(f"\n{'-' * 10} \033[96mSummary\033[0m {'-' * 10}")
        print(f"Number of messages: {len(message.messages)}")
        print(f"Finish reason: {message.stop_reason}")
        print(f"Total prompt tokens: {total_usage.prompt_tokens}")
        print(f"Total completion tokens: {total_usage.completion_tokens}")
        print(f"Duration: {duration:.2f} seconds")
    elif isinstance(message, TextMessage):
        print(f"\n\033[90m[{now}]\033[0m \033[92m[{source}]\033[0m: {message.content}")
    elif isinstance(message, HandoffMessage):
        print(f"\n\033[90m[{now}]\033[0m \033[96m[{source}]\033[0m: Handoff to {message.target}")
        if message.content:
            print(f"\033[90m  Message: {message.content}\033[0m")
    elif isinstance(message, ToolCallRequestEvent):
        for tc in message.content:
            print(f"\n\033[90m[{now}]\033[0m \033[94m[{source}]\033[0m: Tool Call: {tc.name}")
            print(f"\033[90m  Args: {tc.arguments}\033[0m")
    elif isinstance(message, ToolCallExecutionEvent):
        print(f"\n\033[90m[{now}]\033[0m \033[93m[{source}]\033[0m: Tool Result: {[tr.content for tr in message.content]}")
    else:
        print(f"\n\033[90m[{now}]\033[0m \033[91m[{source}]\033[0m: [{type(message).__name__}] {getattr(message, 'content', str(message)[:200])}")
    
    if hasattr(message, "models_usage") and message.models_usage:
        print(f"\033[90m[Prompt tokens: {message.models_usage.prompt_tokens}, Completion tokens: {message.models_usage.completion_tokens}]\033[0m")
