"""State transition hooks for the graph_state pipeline.

Provides a composable hook system that fires before/after each state
execution in the deterministic state machine — zero extra LLM calls.

Unlike the ``graph_agent`` tool middleware (which wraps LLM-driven tool
callables), state hooks operate on ``QueryContext`` at state boundaries,
matching the code-driven architecture of the state machine.

Usage::

    from mmct.video_pipeline.graph_state.hooks import StateHook
    from mmct.video_pipeline.graph_agent.middleware import get_query_context

    class AuthHook(StateHook):
        def applies_to(self, state):
            return state == QueryState.DISCOVER_VIDEOS

        async def after_state(self, state, ctx, next_state):
            allowed = get_query_context().get("allowed_video_ids")
            if allowed and ctx.effective_video_ids:
                ctx.effective_video_ids = [
                    v for v in ctx.effective_video_ids if v in allowed
                ]
            return None

    orchestrator = StateOrchestrator(
        ...,
        state_hooks=[AuthHook()],
    )
    result = await orchestrator.query(
        "...", query_context={"allowed_video_ids": [...]}
    )
"""

from abc import ABC
from typing import Any, Dict, List, Optional

from loguru import logger

from mmct.video_pipeline.graph_state.state_machine import QueryContext, QueryState

_log = logger.bind(component="state:hooks")


class StateHook(ABC):
    """Abstract base class for state transition hooks.

    Subclass and override ``before_state`` and/or ``after_state``
    to intercept state execution.  Override ``applies_to`` to restrict
    which states the hook acts on.

    Hooks are **pure Python** and add zero LLM calls.
    """

    def applies_to(self, state: QueryState) -> bool:
        """Return ``True`` if this hook should fire for *state*.

        The default implementation applies to **all** states.  Override to
        target specific states::

            def applies_to(self, state):
                return state in {QueryState.DISCOVER_VIDEOS, QueryState.RETRIEVE}
        """
        return True

    async def before_state(
        self, state: QueryState, ctx: QueryContext
    ) -> None:
        """Called **before** the state handler executes.

        Mutate *ctx* in-place to influence the state handler's behavior
        (e.g., constrain ``effective_video_ids`` before RETRIEVE).

        Args:
            state: The state about to execute.
            ctx: The mutable pipeline context.
        """

    async def after_state(
        self, state: QueryState, ctx: QueryContext, next_state: QueryState
    ) -> Optional[QueryState]:
        """Called **after** the state handler executes.

        Return a ``QueryState`` to **override** the transition decided by
        the handler, or ``None`` to keep the original transition.

        Args:
            state: The state that just executed.
            ctx: The mutable pipeline context (post-execution).
            next_state: The next state chosen by the handler.

        Returns:
            A replacement ``QueryState``, or ``None`` to keep *next_state*.
        """
        return None


# ---------------------------------------------------------------------------
# Runner utilities
# ---------------------------------------------------------------------------


async def run_before_hooks(
    hooks: List[StateHook],
    state: QueryState,
    ctx: QueryContext,
) -> None:
    """Run applicable ``before_state`` hooks in order (first → last).

    Errors are logged but do not block state execution.
    """
    for hook in hooks:
        if not hook.applies_to(state):
            continue
        try:
            await hook.before_state(state, ctx)
        except Exception:
            _log.exception(
                f"StateHook {type(hook).__name__}.before_state failed "
                f"for state {state.name}"
            )


async def run_after_hooks(
    hooks: List[StateHook],
    state: QueryState,
    ctx: QueryContext,
    next_state: QueryState,
) -> QueryState:
    """Run applicable ``after_state`` hooks in reverse order (last → first).

    If any hook returns a non-``None`` ``QueryState``, the last such
    override wins (consistent with middleware unwinding order).

    Errors are logged but do not block pipeline progression.

    Returns:
        The (possibly overridden) next state.
    """
    effective_next = next_state
    for hook in reversed(hooks):
        if not hook.applies_to(state):
            continue
        try:
            override = await hook.after_state(state, ctx, effective_next)
            if override is not None:
                _log.info(
                    f"StateHook {type(hook).__name__} overrode transition "
                    f"{state.name} → {effective_next.name} to {override.name}"
                )
                effective_next = override
        except Exception:
            _log.exception(
                f"StateHook {type(hook).__name__}.after_state failed "
                f"for state {state.name}"
            )
    return effective_next
