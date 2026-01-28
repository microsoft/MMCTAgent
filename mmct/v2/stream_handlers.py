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


_TERMINATE_STRING = "TERMINATE"
_MAX_MESSAGES_TO_SEARCH = 3  # Maximum messages to search backwards for JSON


def _extract_json_content_from_messages(messages: list, max_search: int = _MAX_MESSAGES_TO_SEARCH) -> str | None:
    """
    Search backwards through recent messages to find JSON content.
    
    The planner sometimes sends JSON and TERMINATE in separate messages,
    or includes preamble text before the JSON block.
    This function searches backwards (up to max_search messages) to find
    a message containing valid JSON content and extracts just the JSON.
    
    Args:
        messages: List of messages from the task result
        max_search: Maximum number of messages to search backwards
        
    Returns:
        The JSON content string if found, None otherwise
    """
    if not messages:
        return None
    
    # Search backwards through the last N messages
    search_range = min(max_search, len(messages))
    
    for i in range(1, search_range + 1):
        msg = messages[-i]
        content = getattr(msg, 'content', '')
        if not content or not isinstance(content, str):
            continue
        
        cleaned = content.strip()
        
        # Check for ```json code block (may have text before/after)
        if '```json' in cleaned:
            # Extract content between ```json and ```
            start_idx = cleaned.find('```json') + len('```json')
            end_idx = cleaned.find('```', start_idx)
            if end_idx != -1:
                json_str = cleaned[start_idx:end_idx].strip()
                return json_str
        
        # Check for raw JSON object (starts with { somewhere in the content)
        brace_idx = cleaned.find('{')
        if brace_idx != -1:
            # Find the matching closing brace
            potential_json = cleaned[brace_idx:]
            # Remove any TERMINATE at the end
            potential_json = potential_json.rstrip().rstrip(_TERMINATE_STRING).rstrip()
            if potential_json.endswith('}'):
                return potential_json
    
    return None


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
        
        # Search backwards through messages to find JSON content
        json_content = _extract_json_content_from_messages(message.messages)
        if json_content is None:
            final_content = message.messages[-1].content if message.messages else ""
        else:
            final_content = json_content
        
        # Try to parse JSON from final content
        parsed_content = final_content
        try:
            # The extraction function already cleans the content, just parse it
            parsed_content = json.loads(final_content)
        except (json.JSONDecodeError, AttributeError, TypeError):
            # If we couldn't find JSON and couldn't parse, return error
            if json_content is None:
                parsed_content = {
                    "response": "Error: Could not find valid JSON response in agent messages",
                    "answer_found": False,
                    "sources": []
                }
        
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
