"""Unit tests for graph_state state transition hooks."""

import pytest
from typing import Any, Dict, List, Optional

from mmct.video_pipeline.graph_state.hooks import (
    StateHook,
    run_before_hooks,
    run_after_hooks,
)
from mmct.video_pipeline.graph_state.state_machine import QueryContext, QueryState
from mmct.video_pipeline.graph_agent.middleware import (
    get_query_context,
    set_query_context,
    reset_query_context,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides) -> QueryContext:
    """Create a minimal QueryContext for testing."""
    defaults = {"query": "test query", "request_id": "test"}
    defaults.update(overrides)
    return QueryContext(**defaults)


class RecordingHook(StateHook):
    """Records hook invocations for assertion."""

    def __init__(self, name: str = "R"):
        self.name = name
        self.before_calls: List[str] = []
        self.after_calls: List[str] = []

    async def before_state(self, state, ctx):
        self.before_calls.append(f"{self.name}:before:{state.name}")

    async def after_state(self, state, ctx, next_state):
        self.after_calls.append(f"{self.name}:after:{state.name}→{next_state.name}")
        return None


class FilteredHook(StateHook):
    """Only applies to specific states."""

    def __init__(self, states):
        self._states = states
        self.before_calls = []

    def applies_to(self, state):
        return state in self._states

    async def before_state(self, state, ctx):
        self.before_calls.append(state.name)


class OverrideHook(StateHook):
    """Overrides next_state in after_state."""

    def __init__(self, override_to: QueryState):
        self._override_to = override_to

    async def after_state(self, state, ctx, next_state):
        return self._override_to


class ContextMutatingHook(StateHook):
    """Mutates ctx in before_state."""

    def applies_to(self, state):
        return state == QueryState.RETRIEVE

    async def before_state(self, state, ctx):
        ctx.effective_video_ids = ["vid_allowed"]


class ErrorBeforeHook(StateHook):
    """Raises in before_state."""

    async def before_state(self, state, ctx):
        raise RuntimeError("before hook error")


class ErrorAfterHook(StateHook):
    """Raises in after_state."""

    async def after_state(self, state, ctx, next_state):
        raise RuntimeError("after hook error")


class ContextReadingHook(StateHook):
    """Reads query_context in before_state."""

    last_user_id: str = ""

    async def before_state(self, state, ctx):
        qctx = get_query_context()
        self.last_user_id = qctx.get("user_id", "anonymous")


# ---------------------------------------------------------------------------
# Tests: run_before_hooks
# ---------------------------------------------------------------------------


class TestRunBeforeHooks:
    @pytest.mark.asyncio
    async def test_no_hooks(self):
        """No hooks is a no-op."""
        ctx = _make_ctx()
        await run_before_hooks([], QueryState.RETRIEVE, ctx)

    @pytest.mark.asyncio
    async def test_ordering_first_to_last(self):
        """Before hooks fire first → last."""
        h1 = RecordingHook("A")
        h2 = RecordingHook("B")
        ctx = _make_ctx()

        await run_before_hooks([h1, h2], QueryState.PLAN, ctx)

        assert h1.before_calls == ["A:before:PLAN"]
        assert h2.before_calls == ["B:before:PLAN"]
        # A was called first (list order)
        assert h1.before_calls[0] < h2.before_calls[0]  # lexicographic, A < B

    @pytest.mark.asyncio
    async def test_applies_to_filtering(self):
        """Hooks that don't apply are skipped."""
        h = FilteredHook({QueryState.RETRIEVE})
        ctx = _make_ctx()

        await run_before_hooks([h], QueryState.PLAN, ctx)
        assert h.before_calls == []

        await run_before_hooks([h], QueryState.RETRIEVE, ctx)
        assert h.before_calls == ["RETRIEVE"]

    @pytest.mark.asyncio
    async def test_ctx_mutation(self):
        """Before hooks can mutate QueryContext."""
        h = ContextMutatingHook()
        ctx = _make_ctx(effective_video_ids=["vid_a", "vid_b"])

        await run_before_hooks([h], QueryState.RETRIEVE, ctx)
        assert ctx.effective_video_ids == ["vid_allowed"]

    @pytest.mark.asyncio
    async def test_error_doesnt_crash(self):
        """Before hook errors are logged but don't propagate."""
        h_err = ErrorBeforeHook()
        h_ok = RecordingHook("OK")
        ctx = _make_ctx()

        await run_before_hooks([h_err, h_ok], QueryState.PLAN, ctx)
        # h_ok still ran despite h_err raising
        assert h_ok.before_calls == ["OK:before:PLAN"]


# ---------------------------------------------------------------------------
# Tests: run_after_hooks
# ---------------------------------------------------------------------------


class TestRunAfterHooks:
    @pytest.mark.asyncio
    async def test_no_hooks(self):
        """No hooks returns original next_state."""
        result = await run_after_hooks(
            [], QueryState.RETRIEVE, _make_ctx(), QueryState.CHECK_EVIDENCE
        )
        assert result == QueryState.CHECK_EVIDENCE

    @pytest.mark.asyncio
    async def test_ordering_last_to_first(self):
        """After hooks fire last → first."""
        h1 = RecordingHook("A")
        h2 = RecordingHook("B")
        ctx = _make_ctx()

        await run_after_hooks(
            [h1, h2], QueryState.PLAN, ctx, QueryState.VALIDATE_PLAN
        )

        # B should appear first (reversed order)
        assert h2.after_calls == ["B:after:PLAN→VALIDATE_PLAN"]
        assert h1.after_calls == ["A:after:PLAN→VALIDATE_PLAN"]

    @pytest.mark.asyncio
    async def test_no_override_returns_original(self):
        """Hooks returning None preserve original next_state."""
        h = RecordingHook()
        result = await run_after_hooks(
            [h], QueryState.RETRIEVE, _make_ctx(), QueryState.CHECK_EVIDENCE
        )
        assert result == QueryState.CHECK_EVIDENCE

    @pytest.mark.asyncio
    async def test_override_next_state(self):
        """Hook can override the next state."""
        h = OverrideHook(QueryState.ERROR)
        result = await run_after_hooks(
            [h], QueryState.RETRIEVE, _make_ctx(), QueryState.CHECK_EVIDENCE
        )
        assert result == QueryState.ERROR

    @pytest.mark.asyncio
    async def test_last_override_wins(self):
        """When multiple hooks override, the last one processed (first in list due to reversal) wins."""
        h1 = OverrideHook(QueryState.ERROR)
        h2 = OverrideHook(QueryState.SYNTHESIZE)
        # After hooks run reversed: h2, then h1
        # h2 sets SYNTHESIZE, then h1 sees SYNTHESIZE and overrides to ERROR
        result = await run_after_hooks(
            [h1, h2], QueryState.RETRIEVE, _make_ctx(), QueryState.CHECK_EVIDENCE
        )
        assert result == QueryState.ERROR

    @pytest.mark.asyncio
    async def test_applies_to_filtering(self):
        """Hooks that don't apply to the state are skipped."""

        class SelectiveOverride(StateHook):
            def applies_to(self, state):
                return state == QueryState.DISCOVER_VIDEOS

            async def after_state(self, state, ctx, next_state):
                return QueryState.ERROR

        h = SelectiveOverride()
        result = await run_after_hooks(
            [h], QueryState.RETRIEVE, _make_ctx(), QueryState.CHECK_EVIDENCE
        )
        assert result == QueryState.CHECK_EVIDENCE  # hook didn't apply

    @pytest.mark.asyncio
    async def test_error_doesnt_crash(self):
        """After hook errors are logged but don't propagate."""
        h_err = ErrorAfterHook()
        h_ok = RecordingHook("OK")
        ctx = _make_ctx()

        result = await run_after_hooks(
            [h_ok, h_err], QueryState.PLAN, ctx, QueryState.VALIDATE_PLAN
        )
        assert result == QueryState.VALIDATE_PLAN
        assert h_ok.after_calls == ["OK:after:PLAN→VALIDATE_PLAN"]

    @pytest.mark.asyncio
    async def test_ctx_mutation_in_after(self):
        """After hooks can mutate QueryContext."""

        class EvidenceFilter(StateHook):
            def applies_to(self, state):
                return state == QueryState.RETRIEVE

            async def after_state(self, state, ctx, next_state):
                ctx.evidence = [e for e in ctx.evidence if e.get("score", 0) > 0.5]
                return None

        h = EvidenceFilter()
        ctx = _make_ctx()
        ctx.evidence = [
            {"node_id": "a", "score": 0.8},
            {"node_id": "b", "score": 0.3},
            {"node_id": "c", "score": 0.9},
        ]

        await run_after_hooks([h], QueryState.RETRIEVE, ctx, QueryState.CHECK_EVIDENCE)
        assert len(ctx.evidence) == 2
        assert {e["node_id"] for e in ctx.evidence} == {"a", "c"}


# ---------------------------------------------------------------------------
# Tests: query context integration
# ---------------------------------------------------------------------------


class TestQueryContext:
    @pytest.mark.asyncio
    async def test_hook_reads_query_context(self):
        """Hook can read per-query context via get_query_context()."""
        h = ContextReadingHook()
        ctx = _make_ctx()

        token = set_query_context({"user_id": "test_user"})
        try:
            await run_before_hooks([h], QueryState.PLAN, ctx)
            assert h.last_user_id == "test_user"
        finally:
            reset_query_context(token)

    @pytest.mark.asyncio
    async def test_default_context_is_empty(self):
        """Without set_query_context, get_query_context returns {}."""
        h = ContextReadingHook()
        ctx = _make_ctx()

        await run_before_hooks([h], QueryState.PLAN, ctx)
        assert h.last_user_id == "anonymous"


# ---------------------------------------------------------------------------
# Tests: full before + after integration
# ---------------------------------------------------------------------------


class TestFullHookCycle:
    @pytest.mark.asyncio
    async def test_before_and_after_both_fire(self):
        """Both before and after hooks fire for the same state."""
        h = RecordingHook("X")
        ctx = _make_ctx()

        await run_before_hooks([h], QueryState.RETRIEVE, ctx)
        next_state = await run_after_hooks(
            [h], QueryState.RETRIEVE, ctx, QueryState.CHECK_EVIDENCE
        )

        assert h.before_calls == ["X:before:RETRIEVE"]
        assert h.after_calls == ["X:after:RETRIEVE→CHECK_EVIDENCE"]
        assert next_state == QueryState.CHECK_EVIDENCE

    @pytest.mark.asyncio
    async def test_multiple_hooks_full_cycle(self):
        """Multiple hooks: before first→last, after last→first."""
        h1 = RecordingHook("A")
        h2 = RecordingHook("B")
        h3 = RecordingHook("C")
        hooks = [h1, h2, h3]
        ctx = _make_ctx()

        await run_before_hooks(hooks, QueryState.PLAN, ctx)
        await run_after_hooks(hooks, QueryState.PLAN, ctx, QueryState.VALIDATE_PLAN)

        # Before: A, B, C (in order)
        assert h1.before_calls == ["A:before:PLAN"]
        assert h2.before_calls == ["B:before:PLAN"]
        assert h3.before_calls == ["C:before:PLAN"]

        # After: C, B, A (reversed)
        assert h3.after_calls == ["C:after:PLAN→VALIDATE_PLAN"]
        assert h2.after_calls == ["B:after:PLAN→VALIDATE_PLAN"]
        assert h1.after_calls == ["A:after:PLAN→VALIDATE_PLAN"]
