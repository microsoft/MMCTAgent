from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from mmct.acl import (
    AccessCheckResult,
    ACLFilter,
    UserIdentifierContext,
    get_user_identifier_context,
    user_identifier_scope,
    wrap_find_relevant_videos,
    wrap_get_video_overview,
    wrap_search_graph,
    wrap_search_keyframes,
    wrap_traverse_graph,
)


# ---------------------------------------------------------------------------
# request_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_user_identifier_scope_round_trip():
    assert get_user_identifier_context() is None
    async with user_identifier_scope({"user": "alice"}):
        assert get_user_identifier_context() == {"user": "alice"}
    assert get_user_identifier_context() is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_user_identifier_scope_nested_unwinds_lifo():
    async with user_identifier_scope({"user": "alice"}):
        async with user_identifier_scope({"user": "bob"}):
            assert get_user_identifier_context() == {"user": "bob"}
        assert get_user_identifier_context() == {"user": "alice"}
    assert get_user_identifier_context() is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_user_identifier_scope_none_is_noop():
    async with user_identifier_scope({"user": "alice"}):
        async with user_identifier_scope(None):
            # None should not clobber the outer scope.
            assert get_user_identifier_context() == {"user": "alice"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_user_identifier_scope_isolates_concurrent_tasks():
    """Two interleaved tasks under different scopes must not see each other."""
    import asyncio

    seen: list[tuple[str, dict | None]] = []

    async def worker(name: str, ctx: dict, delay: float):
        async with user_identifier_scope(ctx):
            await asyncio.sleep(delay)
            seen.append((name, get_user_identifier_context()))

    await asyncio.gather(
        worker("A", {"user": "alice"}, 0.01),
        worker("B", {"user": "bob"}, 0.005),
        worker("C", {"user": "carol"}, 0.0),
    )

    by_name = {name: ctx for name, ctx in seen}
    assert by_name == {
        "A": {"user": "alice"},
        "B": {"user": "bob"},
        "C": {"user": "carol"},
    }


# ---------------------------------------------------------------------------
# ACLFilter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_invokes_callback_with_user_ctx():
    received: list[tuple[list[str], dict]] = []

    async def cb(video_ids: list[str], user_ctx: UserIdentifierContext):
        received.append((list(video_ids), dict(user_ctx)))
        return AccessCheckResult(access_allowed=["a"], access_denied=["b"])

    flt = ACLFilter(cb)
    allowed = await flt.filter_video_ids(["a", "b"], {"user": "alice"})
    assert allowed == {"a"}
    assert received == [(["a", "b"], {"user": "alice"})]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_empty_input_returns_empty_set():
    cb = AsyncMock()
    flt = ACLFilter(cb)
    assert await flt.filter_video_ids([], {"user": "alice"}) == set()
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_filter_callback_exception_propagates_raw_filter():
    """ACLFilter does NOT swallow callback exceptions.

    Fail-closed wrapping is the wrapper layer's responsibility (via
    _apply_filter_fail_closed).
    """
    from mmct.acl import GraphAuthenticationError

    async def cb(video_ids, user_ctx):
        raise GraphAuthenticationError("bad token")

    flt = ACLFilter(cb)
    with pytest.raises(GraphAuthenticationError):
        await flt.filter_video_ids(["a"], {"user": "alice"})


# ---------------------------------------------------------------------------
# Wrapper helpers — discovery tools
# ---------------------------------------------------------------------------


def _make_filter(allowed: list[str], denied: list[str] | None = None,
                 check_failed: list[str] | None = None) -> ACLFilter:
    async def cb(video_ids, user_ctx):
        return AccessCheckResult(
            access_allowed=allowed,
            access_denied=denied or [],
            check_failed=check_failed or [],
        )
    return ACLFilter(cb)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_find_relevant_videos_filters_and_recomputes_total():
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

    flt = _make_filter(allowed=["a"], denied=["b"], check_failed=["c"])
    wrapped = wrap_find_relevant_videos(underlying, flt)

    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(query="x"))

    assert [e["video_id"] for e in out["relevant_videos"]] == ["a"]
    assert out["total_found"] == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_search_graph_filters_every_granularity():
    async def underlying(**kwargs):
        return json.dumps(
            {
                "chapter": [{"id": "ch1", "video_id": "a"}, {"id": "ch2", "video_id": "b"}],
                "topic":   [{"id": "t1",  "video_id": "b"}, {"id": "t2",  "video_id": "a"}],
            }
        )

    flt = _make_filter(allowed=["a"], denied=["b"])
    wrapped = wrap_search_graph(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(query="x"))

    assert [e["video_id"] for e in out["chapter"]] == ["a"]
    assert [e["video_id"] for e in out["topic"]] == ["a"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_traverse_graph_filters_results():
    async def underlying(**kwargs):
        return json.dumps(
            {"results": [{"id": "n1", "video_id": "a"}, {"id": "n2", "video_id": "b"}]}
        )

    flt = _make_filter(allowed=["a"], denied=["b"])
    wrapped = wrap_traverse_graph(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(node_id="x"))

    assert [e["video_id"] for e in out["results"]] == ["a"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_search_keyframes_filters_keyframes():
    async def underlying(**kwargs):
        return json.dumps(
            {"keyframes": [{"id": "k1", "video_id": "a"}, {"id": "k2", "video_id": "b"}]}
        )

    flt = _make_filter(allowed=["a"], denied=["b"])
    wrapped = wrap_search_keyframes(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped())

    assert [e["video_id"] for e in out["keyframes"]] == ["a"]


# ---------------------------------------------------------------------------
# Wrapper helpers — get_video_overview (special semantics)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_get_video_overview_allows_when_video_permitted():
    payload = {"video_id": "a", "nodes": [{"node_id": "n1", "type": "topic"}]}

    async def underlying(**kwargs):
        return json.dumps(payload)

    flt = _make_filter(allowed=["a"])
    wrapped = wrap_get_video_overview(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(video_id="a"))

    assert out == payload


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_get_video_overview_returns_error_when_denied():
    async def underlying(**kwargs):
        return json.dumps({"video_id": "a", "nodes": [{"node_id": "n1"}]})

    flt = _make_filter(allowed=[], denied=["a"])
    wrapped = wrap_get_video_overview(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(video_id="a"))

    assert out == {"error": "access denied for video a"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_get_video_overview_extracts_video_id_from_nested_node():
    """If top-level video_id is missing, fall back to the first node entry."""
    async def underlying(**kwargs):
        return json.dumps({"nodes": [{"node_id": "n1", "video_id": "a"}]})

    flt = _make_filter(allowed=[], denied=["a"])
    wrapped = wrap_get_video_overview(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(video_id="a"))
    assert out == {"error": "access denied for video a"}


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrap_get_video_overview_fail_closed_when_video_id_unextractable():
    """Parseable overview with no video_id anywhere → fail closed, not raw."""
    async def underlying(**kwargs):
        return json.dumps({"summary": "no ids here", "transcripts": ["a", "b"]})

    cb = AsyncMock()
    flt = ACLFilter(cb)
    wrapped = wrap_get_video_overview(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(video_id="x"))
    assert "error" in out
    assert "video_id missing" in out["error"]
    cb.assert_not_called()


# ---------------------------------------------------------------------------
# Generic wrapper passthrough behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrapper_passes_through_on_error_shape():
    async def underlying(**kwargs):
        return json.dumps({"error": "boom"})

    cb = AsyncMock()
    flt = ACLFilter(cb)
    wrapped = wrap_search_graph(underlying, flt)

    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(query="x"))

    assert out == {"error": "boom"}
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrapper_passes_through_on_malformed_json():
    async def underlying(**kwargs):
        return "not json{{"

    cb = AsyncMock()
    flt = ACLFilter(cb)
    wrapped = wrap_find_relevant_videos(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        assert await wrapped(query="x") == "not json{{"
    cb.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrapper_passes_through_when_no_video_ids_present():
    payload = json.dumps({"unrelated": "shape"})

    async def underlying(**kwargs):
        return payload

    cb = AsyncMock()
    flt = ACLFilter(cb)
    wrapped = wrap_find_relevant_videos(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        assert await wrapped(query="x") == payload
    cb.assert_not_called()


# ---------------------------------------------------------------------------
# Defense-in-depth: missing user_identifier_context at filter time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrapper_raises_when_user_ctx_missing():
    """If the pipeline forgot to open user_identifier_scope, fail loudly."""
    async def underlying(**kwargs):
        return json.dumps(
            {"relevant_videos": [{"video_id": "a"}], "total_found": 1}
        )

    flt = _make_filter(allowed=["a"])
    wrapped = wrap_find_relevant_videos(underlying, flt)
    with pytest.raises(ValueError, match="user_identifier_context"):
        await wrapped(query="x")


# ---------------------------------------------------------------------------
# Fail-closed on callback exception (via _apply_filter_fail_closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wrapper_fail_closed_on_callback_exception():
    from mmct.acl import GraphAuthenticationError

    async def underlying(**kwargs):
        return json.dumps(
            {"relevant_videos": [{"video_id": "a"}, {"video_id": "b"}], "total_found": 2}
        )

    async def failing_cb(video_ids, user_ctx):
        raise GraphAuthenticationError("bad token")

    flt = ACLFilter(failing_cb)
    wrapped = wrap_find_relevant_videos(underlying, flt)
    async with user_identifier_scope({"user": "alice"}):
        out = json.loads(await wrapped(query="x"))
    assert out["relevant_videos"] == []
    assert out["total_found"] == 0


# ---------------------------------------------------------------------------
# Pipeline-level fail-fast (uses minimal env stub since heavy deps absent)
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_settings_cache():
    """Clear get_settings's lru_cache so env-var changes take effect."""
    from config.provider_config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


try:
    from mmct.video_pipeline.query_pipeline import VideoQueryPipeline as _VQP_PROBE
    _PIPELINE_AVAILABLE = True
except Exception:  # pragma: no cover — env-dependent
    _VQP_PROBE = None
    _PIPELINE_AVAILABLE = False

_skip_no_pipeline = pytest.mark.skipif(
    not _PIPELINE_AVAILABLE,
    reason="VideoQueryPipeline import requires heavy ML deps",
)

try:
    from mmct.video_pipeline.graph_state import orchestrator as _state_orch_mod
    _STATE_ORCH_AVAILABLE = True
except Exception:  # pragma: no cover — env-dependent
    _state_orch_mod = None
    _STATE_ORCH_AVAILABLE = False

_skip_no_state = pytest.mark.skipif(
    not _STATE_ORCH_AVAILABLE,
    reason="StateOrchestrator import requires heavy ML deps",
)


@_skip_no_pipeline
@pytest.mark.unit
def test_pipeline_construct_without_callback_raises_when_enabled(
    monkeypatch, reset_settings_cache
):
    """ACL_ENABLED=true + no acl_callback at __init__ → fail-fast."""
    from mmct.utils.error_handler import ConfigurationException
    from mmct.video_pipeline.query_pipeline import VideoQueryPipeline, QueryPipelineMode

    monkeypatch.setenv("ACL_ENABLED", "true")
    with pytest.raises(ConfigurationException, match="acl_callback"):
        VideoQueryPipeline(
            mode=QueryPipelineMode.GRAPH_STATE,
            model_client=object(),
            neo4j_provider=object(),
        )


# ---------------------------------------------------------------------------
# StateOrchestrator integration with ContextVar
# ---------------------------------------------------------------------------


def _make_state_orch(monkeypatch, acl_callback=None):
    so_mod = _state_orch_mod
    monkeypatch.setattr(so_mod, "RetrievalExecutor", lambda *a, **kw: object(), raising=True)
    monkeypatch.setattr(so_mod, "VideoDiscoveryExecutor", lambda *a, **kw: object(), raising=True)
    monkeypatch.setattr(so_mod, "ImageAnalysisExecutor", lambda *a, **kw: object(), raising=True)
    monkeypatch.setattr(so_mod, "PlannerAgent", lambda *a, **kw: object(), raising=True)
    monkeypatch.setattr(so_mod, "CriticAgent", lambda *a, **kw: object(), raising=True)
    return so_mod.StateOrchestrator(
        model_client=object(),
        neo4j_provider=object(),
        acl_callback=acl_callback,
    )


@_skip_no_state
@pytest.mark.unit
def test_state_orchestrator_no_callback_no_filter(monkeypatch):
    orch = _make_state_orch(monkeypatch)
    assert orch._acl_filter is None


@_skip_no_pipeline
@pytest.mark.unit
def test_pipeline_toggle_off_ignores_supplied_callback(
    monkeypatch, reset_settings_cache
):
    """ACL_ENABLED=false + callback supplied → orchestrator gets None.

    The env var is the single source of truth; a callback handed in by a
    well-meaning caller while the toggle is off must be silently ignored
    (no filtering, no per-request user_ctx requirement).
    """
    from mmct.video_pipeline.query_pipeline import VideoQueryPipeline, QueryPipelineMode

    monkeypatch.setenv("ACL_ENABLED", "false")

    async def cb(video_ids, user_ctx):
        return AccessCheckResult(access_allowed=list(video_ids))

    pipeline = VideoQueryPipeline(
        mode=QueryPipelineMode.GRAPH_STATE,
        model_client=object(),
        neo4j_provider=object(),
        acl_callback=cb,
    )
    assert pipeline._orchestrator._acl_filter is None


@_skip_no_state
@pytest.mark.unit
def test_state_orchestrator_with_callback_constructs_filter(monkeypatch):
    async def cb(video_ids, user_ctx):
        return AccessCheckResult(access_allowed=list(video_ids))
    orch = _make_state_orch(monkeypatch, acl_callback=cb)
    assert orch._acl_filter is not None


@_skip_no_state
@pytest.mark.asyncio
@pytest.mark.unit
async def test_state_discover_videos_reads_user_ctx_from_contextvar(monkeypatch):
    received: list[dict] = []

    async def cb(video_ids, user_ctx):
        received.append(dict(user_ctx))
        return AccessCheckResult(
            access_allowed=["a"], access_denied=["b"], check_failed=["c"]
        )

    orch = _make_state_orch(monkeypatch, acl_callback=cb)

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
    async with user_identifier_scope({"user": "alice"}):
        next_state = await orch._state_discover_videos(qctx)

    assert received == [{"user": "alice"}]
    assert qctx.effective_video_ids == ["a"]
    assert next_state == QueryState.RETRIEVE


@_skip_no_state
@pytest.mark.asyncio
@pytest.mark.unit
async def test_state_discover_videos_raises_when_user_ctx_missing(monkeypatch):
    """ACL on, but user_identifier_scope wasn't opened — defensive raise."""
    async def cb(video_ids, user_ctx):
        return AccessCheckResult(access_allowed=list(video_ids))
    orch = _make_state_orch(monkeypatch, acl_callback=cb)

    class _StubDiscovery:
        async def discover(self, query, limit):
            return [("a", 0.9, "title a")]

    orch._discovery = _StubDiscovery()

    from mmct.video_pipeline.graph_state.state_machine import QueryContext

    qctx = QueryContext(query="x", request_id="rid-2")
    with pytest.raises(ValueError, match="user_identifier_context"):
        await orch._state_discover_videos(qctx)


# ---------------------------------------------------------------------------
# VideoAgent: all 5 tools wrapped when callback present
# ---------------------------------------------------------------------------


try:
    from mmct.video_pipeline.graph_agent.agents import video_agent as _video_agent_mod
    _VIDEO_AGENT_AVAILABLE = True
except Exception:  # pragma: no cover — env-dependent
    _video_agent_mod = None
    _VIDEO_AGENT_AVAILABLE = False

_skip_no_va = pytest.mark.skipif(
    not _VIDEO_AGENT_AVAILABLE,
    reason="VideoAgent import requires heavy ML deps (autogen, modelscope)",
)


@_skip_no_va
@pytest.mark.unit
def test_video_agent_no_callback_leaves_tools_unwrapped():
    from unittest.mock import patch
    va_mod = _video_agent_mod
    with patch.object(va_mod.AssistantAgent, "__init__", return_value=None):
        agent = va_mod.VideoAgent(model_client=object(), neo4j_provider=object())
    assert agent._acl_filter is None
    for tool in agent.tools:
        # Bound method has __self__; wrappers are plain functions.
        assert hasattr(tool, "__self__"), f"{tool} should be a bound method"


@_skip_no_va
@pytest.mark.unit
def test_video_agent_with_callback_wraps_all_five_tools():
    from unittest.mock import patch
    va_mod = _video_agent_mod

    async def cb(video_ids, user_ctx):
        return AccessCheckResult(access_allowed=list(video_ids))

    with patch.object(va_mod.AssistantAgent, "__init__", return_value=None):
        agent = va_mod.VideoAgent(
            model_client=object(),
            neo4j_provider=object(),
            acl_callback=cb,
        )
    assert agent._acl_filter is not None
    expected_names = {
        "get_video_overview", "search_graph", "traverse_graph",
        "search_keyframes", "find_relevant_videos",
    }
    tool_names = {getattr(t, "__name__", "") for t in agent.tools}
    assert tool_names == expected_names
    # All five should be wrappers (plain functions), not bound methods.
    for tool in agent.tools:
        assert not hasattr(tool, "__self__"), f"{tool} should be wrapped"
