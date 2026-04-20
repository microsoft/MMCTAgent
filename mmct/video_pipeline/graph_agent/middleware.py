"""Tool middleware for the graph_agent pipeline.

Provides a composable middleware system that wraps agent tool calls
with before/after hooks — zero extra LLM calls.  Middleware is pure
Python and invisible to the LLM; ``functools.wraps`` preserves the
original tool signature so AutoGen schema generation is unaffected.

Usage::

    from mmct.video_pipeline.graph_agent.middleware import (
        ToolMiddleware,
        apply_middleware,
        get_query_context,
    )

    class AuthMiddleware(ToolMiddleware):
        async def before_tool_call(self, tool_name, kwargs):
            ctx = get_query_context()
            allowed = ctx.get("allowed_video_ids", [])
            if "video_ids" in kwargs and kwargs["video_ids"] is not None:
                kwargs["video_ids"] = [v for v in kwargs["video_ids"] if v in allowed]
            return kwargs

    orchestrator = GraphOrchestrator(
        ...,
        tool_middleware=[AuthMiddleware()],
    )
    result = await orchestrator.query("...", query_context={"allowed_video_ids": [...]})
"""

import contextvars
import functools
import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from loguru import logger

_log = logger.bind(component="middleware")

# ---------------------------------------------------------------------------
# Per-query context via contextvars (async-safe, no tool signature pollution)
# ---------------------------------------------------------------------------

_query_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "query_context", default={}
)


def get_query_context() -> Dict[str, Any]:
    """Return the current per-query context dictionary.

    Call this from within a ``ToolMiddleware`` hook to access request-scoped
    data (e.g. ``user_id``, ``roles``) set by the orchestrator.
    """
    return _query_context.get()


def set_query_context(ctx: Dict[str, Any]) -> contextvars.Token:
    """Set the per-query context.  Returns a token for ``reset_query_context``.

    Typically called by the orchestrator before running the swarm.
    """
    return _query_context.set(ctx)


def reset_query_context(token: contextvars.Token) -> None:
    """Reset the per-query context to its previous value."""
    _query_context.reset(token)


# ---------------------------------------------------------------------------
# ToolMiddleware ABC
# ---------------------------------------------------------------------------


class ToolMiddleware(ABC):
    """Abstract base class for tool middleware.

    Subclass and override ``before_tool_call`` and/or ``after_tool_call``
    to intercept tool invocations.  Override ``applies_to`` to restrict
    which tools the middleware acts on.

    Middleware hooks are **pure Python** and add zero LLM calls.
    """

    def applies_to(self, tool_name: str) -> bool:
        """Return ``True`` if this middleware should wrap *tool_name*.

        The default implementation applies to **all** tools.  Override to
        target specific tools::

            def applies_to(self, tool_name):
                return tool_name in {"search_graph", "find_relevant_videos"}
        """
        return True

    async def before_tool_call(
        self, tool_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Called **before** the tool executes.

        Receives the keyword arguments the LLM chose for this tool call.
        Return a (possibly modified) ``kwargs`` dict that will be forwarded
        to the actual tool function.

        Args:
            tool_name: Name of the tool about to be called.
            kwargs: Keyword arguments destined for the tool.

        Returns:
            The (possibly modified) keyword arguments dictionary.
        """
        return kwargs

    async def after_tool_call(
        self, tool_name: str, result: str, kwargs: Dict[str, Any]
    ) -> str:
        """Called **after** the tool executes.

        Receives the string result returned by the tool.  Return a
        (possibly modified) result string that will be sent back to the LLM.

        Args:
            tool_name: Name of the tool that was called.
            result: The string result from the tool.
            kwargs: The keyword arguments that were passed to the tool
                (after any ``before_tool_call`` modifications).

        Returns:
            The (possibly modified) result string.
        """
        return result


# ---------------------------------------------------------------------------
# Wrapping utility
# ---------------------------------------------------------------------------


def apply_middleware(
    tool_func: Callable,
    middlewares: List[ToolMiddleware],
) -> Callable:
    """Wrap *tool_func* with all applicable middlewares.

    Before hooks run **first → last**; after hooks run **last → first**
    (standard middleware unwinding order).

    Uses ``functools.wraps`` so that the wrapper preserves the original
    function's ``__name__``, ``__doc__``, ``__annotations__``, and
    ``__module__`` — ensuring AutoGen schema generation sees the
    original tool signature.

    Args:
        tool_func: The original tool callable (async function).
        middlewares: Flat list of ``ToolMiddleware`` instances.

    Returns:
        A wrapped callable with the same signature as *tool_func*,
        or *tool_func* unchanged if no middleware applies.
    """
    tool_name = getattr(tool_func, "__name__", str(tool_func))
    applicable = [m for m in middlewares if m.applies_to(tool_name)]

    if not applicable:
        return tool_func

    @functools.wraps(tool_func)
    async def wrapper(**kwargs: Any) -> str:
        # Before hooks: first → last
        modified_kwargs = kwargs
        for mw in applicable:
            try:
                modified_kwargs = await mw.before_tool_call(tool_name, modified_kwargs)
            except Exception:
                _log.exception(
                    f"Middleware {type(mw).__name__}.before_tool_call failed "
                    f"for tool '{tool_name}'"
                )

        # Execute the original tool
        result = await tool_func(**modified_kwargs)

        # After hooks: last → first
        for mw in reversed(applicable):
            try:
                result = await mw.after_tool_call(tool_name, result, modified_kwargs)
            except Exception:
                _log.exception(
                    f"Middleware {type(mw).__name__}.after_tool_call failed "
                    f"for tool '{tool_name}'"
                )

        return result

    _log.debug(
        f"Wrapped '{tool_name}' with {len(applicable)} middleware(s): "
        f"{[type(m).__name__ for m in applicable]}"
    )
    return wrapper
