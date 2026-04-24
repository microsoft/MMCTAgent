from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from mmct.acl import (
    AccessCheckResult,
    ACLContext,
    ACLFilter,
    VideoIdentifier,
)
from mmct.acl.filter import wrap_find_relevant_videos, wrap_search_graph

# VideoAgent lives in a package whose siblings pull in heavy ML deps
# (autogen, modelscope, numpy, ...). Skip the VideoAgent integration tests
# when those aren't importable, so the ACL primitives remain testable in a
# minimal env.
try:
    from mmct.video_pipeline.graph_agent.agents import video_agent as _video_agent_mod
    _VIDEO_AGENT_AVAILABLE = True
except Exception:  # pragma: no cover — env-dependent
    _video_agent_mod = None
    _VIDEO_AGENT_AVAILABLE = False

try:
    from mmct.video_pipeline.graph_state import orchestrator as _state_orch_mod
    _STATE_ORCH_AVAILABLE = True
except Exception:  # pragma: no cover — env-dependent
    _state_orch_mod = None
    _STATE_ORCH_AVAILABLE = False


def _vid(video_id: str) -> VideoIdentifier:
    return VideoIdentifier(video_id=video_id, drive_id=f"d-{video_id}", item_id=f"i-{video_id}")


# ---------------------------------------------------------------------------
# ACLContext validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_context_requires_callback_or_default_pair():
    with pytest.raises(ValueError):
        ACLContext()
    with pytest.raises(ValueError):
        ACLContext(graph_token="tok")
    async def _lookup(ids):  # pragma: no cover — construction-only
        return []
    with pytest.raises(ValueError):
        ACLContext(video_identifier_lookup=_lookup)


@pytest.mark.unit
def test_context_valid_with_custom_callback():
    async def _cb(ids):  # pragma: no cover — construction-only
        return AccessCheckResult()
    ctx = ACLContext(access_check_callback=_cb)
    assert ctx.access_check_callback is _cb


@pytest.mark.unit
def test_context_valid_with_default_pair():
    async def _lookup(ids):  # pragma: no cover — construction-only
        return []
    ctx = ACLContext(graph_token="tok", video_identifier_lookup=_lookup)
    assert ctx.graph_token == "tok"


# ---------------------------------------------------------------------------
# ACLFilter.filter_video_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_uses_default_callback_via_check_access_to_video_list():
    lookup = AsyncMock(return_value=[_vid("a"), _vid("b"), _vid("c")])
    ctx = ACLContext(graph_token="tok", video_identifier_lookup=lookup)
    flt = ACLFilter(ctx)

    fake_result = AccessCheckResult(
        access_allowed=["a"],
        access_denied=["b"],
        check_failed=["c"],
    )
    with patch(
        "mmct.acl.filter.check_access_to_video_list",
        new=AsyncMock(return_value=fake_result),
    ) as mocked:
        allowed = await flt.filter_video_ids(["a", "b", "c"])

    assert allowed == {"a"}
    lookup.assert_awaited_once_with(["a", "b", "c"])
    mocked.assert_awaited_once()
    args, _ = mocked.call_args
    assert args[0] == "tok"
    assert [v.video_id for v in args[1]] == ["a", "b", "c"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_custom_callback_bypasses_default_adapter():
    received: list[list[str]] = []

    async def custom_cb(ids):
        received.append(list(ids))
        return AccessCheckResult(access_allowed=["a"], access_denied=["b"])

    ctx = ACLContext(access_check_callback=custom_cb)
    flt = ACLFilter(ctx)

    with patch(
        "mmct.acl.filter.check_access_to_video_list",
        new=AsyncMock(),
    ) as mocked_default:
        allowed = await flt.filter_video_ids(["a", "b"])

    assert allowed == {"a"}
    assert received == [["a", "b"]]
    mocked_default.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_empty_input_returns_empty_set():
    cb = AsyncMock()
    ctx = ACLContext(access_check_callback=cb)
    flt = ACLFilter(ctx)
    assert await flt.filter_video_ids([]) == set()
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_callback_exception_propagates():
    """filter_video_ids does NOT swallow callback exceptions on its own.

    Fail-closed wrapping is the caller's responsibility (done by the
    wrappers and orchestrator via _apply_filter_fail_closed).
    """
    from mmct.acl import GraphAuthenticationError

    async def failing_cb(ids):
        raise GraphAuthenticationError("bad token")

    flt = ACLFilter(ACLContext(access_check_callback=failing_cb))
    with pytest.raises(GraphAuthenticationError):
        await flt.filter_video_ids(["a"])


@pytest.mark.unit
def test_context_repr_does_not_leak_graph_token():
    async def _lookup(ids):
        return []

    ctx = ACLContext(graph_token="super-secret-token", video_identifier_lookup=_lookup)
    rendered = repr(ctx)
    assert "super-secret-token" not in rendered


# ---------------------------------------------------------------------------
# VideoAgent integration — toggle off (bypass)
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_settings_cache():
    """Make env-var toggle changes visible to get_settings() for the test.

    In a full-deps env, clear the lru_cache on the real get_settings so it
    re-reads. In a minimal test env where config.provider_config can't be
    imported (azure/autogen transitive deps missing), install a tiny stub
    module that reads the toggles live from os.environ. The orchestrator /
    agent code under test imports get_settings lazily, so the stub is
    picked up at __init__ time.
    """
    import os
    import sys
    import types

    try:
        from config.provider_config import get_settings  # type: ignore
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()
        return
    except Exception:
        pass

    class _StubSettings:
        @property
        def acl_enabled_graph_agent(self) -> bool:
            return os.environ.get("ACL_ENABLED_GRAPH_AGENT", "").lower() == "true"

        @property
        def acl_enabled_graph_state(self) -> bool:
            return os.environ.get("ACL_ENABLED_GRAPH_STATE", "").lower() == "true"

    config_pkg = types.ModuleType("config")
    provider_module = types.ModuleType("config.provider_config")
    provider_module.get_settings = lambda: _StubSettings()
    config_pkg.provider_config = provider_module
    sys.modules["config"] = config_pkg
    sys.modules["config.provider_config"] = provider_module
    try:
        yield
    finally:
        sys.modules.pop("config.provider_config", None)
        sys.modules.pop("config", None)


_skip_no_va = pytest.mark.skipif(
    not _VIDEO_AGENT_AVAILABLE,
    reason="VideoAgent package requires heavy ML deps (autogen, modelscope, ...)",
)


@_skip_no_va
@pytest.mark.unit
def test_video_agent_disabled_leaves_tools_unwrapped(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_AGENT", "false")
    va_mod = _video_agent_mod

    with patch.object(va_mod.AssistantAgent, "__init__", return_value=None):
        agent = va_mod.VideoAgent(model_client=object(), neo4j_provider=object())

    assert agent._acl_filter is None
    # Sanity: discovery tools are present on self.tools as bound methods, not wrappers.
    names = [getattr(t, "__name__", "") for t in agent.tools]
    assert "search_graph" in names
    assert "find_relevant_videos" in names
    # A wrapper would still carry the original name via @wraps, so check it's
    # actually a bound method rather than a plain function.
    for tool in agent.tools:
        assert hasattr(tool, "__self__"), f"{tool} should be a bound method when ACL off"


@_skip_no_va
@pytest.mark.unit
def test_video_agent_enabled_without_context_raises(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_AGENT", "true")
    va_mod = _video_agent_mod
    with pytest.raises(ValueError, match="ACL_ENABLED_GRAPH_AGENT"):
        va_mod.VideoAgent(model_client=object(), neo4j_provider=object())


@_skip_no_va
@pytest.mark.unit
def test_video_agent_enabled_with_custom_callback_initializes(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_AGENT", "true")
    va_mod = _video_agent_mod

    async def cb(ids):
        return AccessCheckResult(access_allowed=list(ids))

    ctx = ACLContext(access_check_callback=cb)
    with patch.object(va_mod.AssistantAgent, "__init__", return_value=None):
        agent = va_mod.VideoAgent(
            model_client=object(),
            neo4j_provider=object(),
            acl_context=ctx,
        )
    assert agent._acl_filter is not None

    # Discovery tools should be wrappers (plain functions, not bound methods).
    tools_by_name = {getattr(t, "__name__", ""): t for t in agent.tools}
    for name in ("search_graph", "find_relevant_videos"):
        t = tools_by_name[name]
        assert not hasattr(t, "__self__"), f"{name} should be wrapped (not bound)"
    # Derivative tools stay bound.
    for name in ("get_video_overview", "traverse_graph", "search_keyframes"):
        t = tools_by_name[name]
        assert hasattr(t, "__self__"), f"{name} should remain bound when ACL on"


# ---------------------------------------------------------------------------
# Wrapper behavior (exercises the actual filtering logic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_find_relevant_videos_wrapper_filters_and_recomputes_total():
    from mmct.acl.filter import wrap_find_relevant_videos as _wrap_find_relevant_videos

    async def underlying(**kwargs):
        return json.dumps(
            {
                "relevant_videos": [
                    {"video_id": "a", "relevance_score": 0.9},
                    {"video_id": "b", "relevance_score": 0.8},
                    {"video_id": "c", "relevance_score": 0.7},
                ],
                "total_found": 3,
            }
        )

    async def cb(ids):
        return AccessCheckResult(
            access_allowed=["a"], access_denied=["b"], check_failed=["c"]
        )

    flt = ACLFilter(ACLContext(access_check_callback=cb))
    wrapped = _wrap_find_relevant_videos(underlying, flt)

    out = json.loads(await wrapped(query="anything"))
    assert [e["video_id"] for e in out["relevant_videos"]] == ["a"]
    assert out["total_found"] == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_search_graph_wrapper_filters_every_granularity():
    from mmct.acl.filter import wrap_search_graph as _wrap_search_graph

    async def underlying(**kwargs):
        return json.dumps(
            {
                "chapter": [
                    {"id": "ch1", "video_id": "a"},
                    {"id": "ch2", "video_id": "b"},
                ],
                "topic": [
                    {"id": "t1", "video_id": "b"},
                    {"id": "t2", "video_id": "a"},
                ],
            }
        )

    async def cb(ids):
        return AccessCheckResult(
            access_allowed=["a"], access_denied=["b"], check_failed=[]
        )

    flt = ACLFilter(ACLContext(access_check_callback=cb))
    wrapped = _wrap_search_graph(underlying, flt)

    out = json.loads(await wrapped(query="x"))
    assert [e["video_id"] for e in out["chapter"]] == ["a"]
    assert [e["video_id"] for e in out["topic"]] == ["a"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_search_graph_wrapper_passes_through_on_error_shape():
    from mmct.acl.filter import wrap_search_graph as _wrap_search_graph

    async def underlying(**kwargs):
        return json.dumps({"error": "boom"})

    cb = AsyncMock()
    flt = ACLFilter(ACLContext(access_check_callback=cb))
    wrapped = _wrap_search_graph(underlying, flt)

    out = json.loads(await wrapped(query="x"))
    assert out == {"error": "boom"}
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_find_relevant_videos_wrapper_malformed_json_passthrough():
    from mmct.acl.filter import wrap_find_relevant_videos as _wrap

    async def underlying(**kwargs):
        return "not valid json{{"

    cb = AsyncMock()
    flt = ACLFilter(ACLContext(access_check_callback=cb))
    wrapped = _wrap(underlying, flt)
    assert await wrapped(query="x") == "not valid json{{"
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_find_relevant_videos_wrapper_missing_key_passthrough():
    from mmct.acl.filter import wrap_find_relevant_videos as _wrap

    payload = json.dumps({"unrelated": "shape"})

    async def underlying(**kwargs):
        return payload

    cb = AsyncMock()
    flt = ACLFilter(ACLContext(access_check_callback=cb))
    wrapped = _wrap(underlying, flt)
    assert await wrapped(query="x") == payload
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrapper_fail_closed_on_callback_exception():
    """When the ACL callback raises, the wrapper drops all video_ids."""
    from mmct.acl import GraphAuthenticationError
    from mmct.acl.filter import wrap_find_relevant_videos as _wrap

    async def underlying(**kwargs):
        return json.dumps(
            {"relevant_videos": [{"video_id": "a"}, {"video_id": "b"}], "total_found": 2}
        )

    async def failing_cb(ids):
        raise GraphAuthenticationError("bad token")

    flt = ACLFilter(ACLContext(access_check_callback=failing_cb))
    wrapped = _wrap(underlying, flt)
    out = json.loads(await wrapped(query="x"))
    assert out["relevant_videos"] == []
    assert out["total_found"] == 0


# ---------------------------------------------------------------------------
# StateOrchestrator integration (Phase 2)
# ---------------------------------------------------------------------------

_skip_no_state = pytest.mark.skipif(
    not _STATE_ORCH_AVAILABLE,
    reason="StateOrchestrator requires autogen/neo4j deps",
)


def _make_state_orch(monkeypatch, acl_context=None):
    """Construct a StateOrchestrator with all non-ACL dependencies stubbed."""
    so_mod = _state_orch_mod
    monkeypatch.setattr(
        so_mod, "RetrievalExecutor", lambda *a, **kw: object(), raising=True
    )
    monkeypatch.setattr(
        so_mod, "VideoDiscoveryExecutor", lambda *a, **kw: object(), raising=True
    )
    monkeypatch.setattr(
        so_mod, "ImageAnalysisExecutor", lambda *a, **kw: object(), raising=True
    )
    monkeypatch.setattr(so_mod, "PlannerAgent", lambda *a, **kw: object(), raising=True)
    monkeypatch.setattr(so_mod, "CriticAgent", lambda *a, **kw: object(), raising=True)
    return so_mod.StateOrchestrator(
        model_client=object(),
        neo4j_provider=object(),
        acl_context=acl_context,
    )


@_skip_no_state
@pytest.mark.unit
def test_state_orchestrator_disabled_has_no_filter(monkeypatch, reset_settings_cache):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_STATE", "false")
    orch = _make_state_orch(monkeypatch)
    assert orch._acl_filter is None


@_skip_no_state
@pytest.mark.unit
def test_state_orchestrator_enabled_without_context_raises(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_STATE", "true")
    with pytest.raises(ValueError, match="ACL_ENABLED_GRAPH_STATE"):
        _make_state_orch(monkeypatch)


@_skip_no_state
@pytest.mark.unit
def test_state_orchestrator_enabled_with_custom_callback_initializes(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_STATE", "true")

    async def cb(ids):
        return AccessCheckResult(access_allowed=list(ids))

    ctx = ACLContext(access_check_callback=cb)
    orch = _make_state_orch(monkeypatch, acl_context=ctx)
    assert orch._acl_filter is not None


@_skip_no_state
@pytest.mark.asyncio
@pytest.mark.unit
async def test_state_discover_videos_filters_ranked(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_STATE", "true")

    cb_calls: list[list[str]] = []

    async def cb(ids):
        cb_calls.append(list(ids))
        return AccessCheckResult(
            access_allowed=["a"], access_denied=["b"], check_failed=["c"]
        )

    acl_ctx = ACLContext(access_check_callback=cb)
    orch = _make_state_orch(monkeypatch, acl_context=acl_ctx)

    class _StubDiscovery:
        async def discover(self, query, limit):
            return [
                ("a", 0.9, "title a"),
                ("b", 0.8, "title b"),
                ("c", 0.7, "title c"),
            ]

    orch._discovery = _StubDiscovery()

    from mmct.video_pipeline.graph_state.state_machine import QueryContext, QueryState

    qctx = QueryContext(query="x", request_id="rid-1")
    next_state = await orch._state_discover_videos(qctx)

    assert cb_calls == [["a", "b", "c"]]
    # Only 'a' is access_allowed; 'b' denied and 'c' check_failed are dropped.
    assert qctx.effective_video_ids == ["a"]
    assert next_state == QueryState.RETRIEVE


@_skip_no_state
@pytest.mark.asyncio
@pytest.mark.unit
async def test_state_discover_videos_disabled_skips_filter(
    monkeypatch, reset_settings_cache
):
    monkeypatch.setenv("ACL_ENABLED_GRAPH_STATE", "false")
    orch = _make_state_orch(monkeypatch)

    class _StubDiscovery:
        async def discover(self, query, limit):
            return [("a", 0.9, "title a"), ("b", 0.8, "title b")]

    orch._discovery = _StubDiscovery()

    from mmct.video_pipeline.graph_state.state_machine import QueryContext

    qctx = QueryContext(query="x", request_id="rid-2")
    await orch._state_discover_videos(qctx)
    # Nothing filtered → both video_ids survive (order preserved).
    assert qctx.effective_video_ids == ["a", "b"]
