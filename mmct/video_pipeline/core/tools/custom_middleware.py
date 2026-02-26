from agent_framework import AgentMiddleware, AgentContext
from agent_framework import ChatMiddleware, ChatContext
from agent_framework import FunctionMiddleware, FunctionInvocationContext
from typing import Callable, Awaitable
from rich.console import Console
from rich.text import Text
import json

console = Console()

def _print_header(text: str, color: str = "cyan"):
    console.print(f"\n[bold {color}]>>> {text}[/]")

def _print_msg(role: str, content: str):
    if role == "user":
        style = "bold cyan"
    elif role == "assistant":
        style = "bold magenta"
    elif role == "system":
        style = "bold yellow"
    elif role == "tool":
        style = "bold green"
    else:
        style = "white"
    
    console.print(Text(f"[{role.upper()}]", style=style), end=" ")
    console.print(content)

class LoggingAgentMiddleware(AgentMiddleware):
    """Simple linear logging for Agent execution."""

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        agent_name = getattr(context.agent, "name", "unknown")
        _print_header(f"AGENT START: {agent_name}", "cyan")
        
        for msg in context.messages:
            _print_msg(getattr(msg, "role", "unknown"), getattr(msg, "text", "") or "")

        await call_next()

        result = context.result
        if result is not None:
            result_msgs = getattr(result, "messages", None)
            if result_msgs:
                _print_header(f"AGENT RESPONSE: {agent_name}", "magenta")
                for msg in result_msgs:
                    _print_msg(getattr(msg, "role", "unknown"), getattr(msg, "text", "") or "")
        
        _print_header(f"AGENT END: {agent_name}", "cyan")

class LoggingChatMiddleware(ChatMiddleware):
    """Simple linear logging for Chat interactions."""

    async def process(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        # We skip redundant logging if AgentMiddleware is already showing it
        await call_next()

class LoggingFunctionMiddleware(FunctionMiddleware):
    """Simple linear logging for Tool calls."""

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        fn_name = context.function.name
        _print_header(f"TOOL CALL: {fn_name}", "green")
        try:
            console.print(f"Arguments: {json.dumps(context.arguments, indent=2)}")
        except Exception:
            console.print(f"Arguments: {str(context.arguments)}")

        try:
            await call_next()
            _print_header(f"TOOL RESULT: {fn_name}", "green")
            try:
                # context.result might be an object that isn't cleanly printable or serializable
                if hasattr(context.result, "model_dump"):
                    console.print(json.dumps(context.result.model_dump(), indent=2))
                else:
                    console.print(str(context.result))
            except Exception:
                console.print(str(context.result))
        except Exception as e:
            _print_header(f"TOOL ERROR: {fn_name}", "red")
            console.print(f"Exception Type: {type(e).__name__}")
            console.print(f"Exception Message: {str(e)}")
            raise e

class TerminationMiddleware(AgentMiddleware):
    """Agent middleware that terminates execution when TERMINATE is seen."""

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        last_message = context.messages[-1] if context.messages else None
        if last_message and last_message.text:
            if "terminate" in last_message.text.lower():
                console.print("[bold red]!!! TERMINATION TRIGGERED !!![/]")
                context.terminate = True
                return

        await call_next()