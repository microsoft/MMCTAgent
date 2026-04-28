"""Tests for the FastAPI video-query ACL plumbing.

Covers the new ``user_identifier_context`` form field on
``POST /video-query`` (route layer behavior). The pipeline is patched
out so these tests stay focused on the HTTP surface.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

# api.main pulls in the image pipeline which transitively imports modelscope.
# Skip cleanly in test envs that don't have those heavy ML deps installed; the
# tests are valid in any env where the FastAPI app can be instantiated.
try:
    from fastapi.testclient import TestClient
    from api.main import app
    _APP_AVAILABLE = True
except Exception:  # pragma: no cover — env-dependent
    app = None
    TestClient = None
    _APP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _APP_AVAILABLE,
    reason="api.main app requires heavy ML deps (modelscope, etc.)",
)


@pytest.fixture
def client():
    return TestClient(app)


def _post_query(client: TestClient, **extra: object):
    data = {
        "query": "what happened in the meeting",
        "mode": "graph_state",
        **extra,
    }
    return client.post("/video-query", data=data)


# ---------------------------------------------------------------------------
# Field is forwarded as a parsed dict to the service layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_user_identifier_context_forwarded_to_service(client):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "sources": []}

    with patch(
        "api.routers.video_query.run_video_query", new=AsyncMock(side_effect=fake_run)
    ):
        resp = _post_query(
            client,
            user_identifier_context='{"email":"alice@example.com","graph_token":"abc"}',
        )

    assert resp.status_code == 200
    assert captured["user_identifier_context"] == {
        "email": "alice@example.com",
        "graph_token": "abc",
    }


@pytest.mark.unit
def test_missing_user_identifier_context_passes_none(client):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok", "sources": []}

    with patch(
        "api.routers.video_query.run_video_query", new=AsyncMock(side_effect=fake_run)
    ):
        resp = _post_query(client)

    assert resp.status_code == 200
    assert captured["user_identifier_context"] is None


# ---------------------------------------------------------------------------
# Malformed input → 400 from the route, NOT 500 from the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    ["not-json{{", '"alice"', "[1,2,3]", "42", "null"],
    ids=["malformed", "string", "array", "number", "null"],
)
def test_invalid_user_identifier_context_returns_400(client, payload):
    with patch(
        "api.routers.video_query.run_video_query",
        new=AsyncMock(return_value={"answer": "n/a"}),
    ) as mock_run:
        resp = _post_query(client, user_identifier_context=payload)

    assert resp.status_code == 400
    assert "JSON object" in resp.json()["detail"]
    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Empty string is treated as None (graceful)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_string_treated_as_missing(client):
    captured: dict = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return {"answer": "ok"}

    with patch(
        "api.routers.video_query.run_video_query", new=AsyncMock(side_effect=fake_run)
    ):
        resp = _post_query(client, user_identifier_context="")

    assert resp.status_code == 200
    assert captured["user_identifier_context"] is None


# ---------------------------------------------------------------------------
# Stream endpoint also accepts the field
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stream_endpoint_forwards_user_identifier_context(client):
    captured: dict = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        if False:
            yield  # generator with no events

    with patch(
        "api.routers.video_query.stream_video_query", new=fake_stream
    ):
        resp = client.post(
            "/video-query/stream",
            data={
                "query": "x",
                "mode": "graph_state",
                "user_identifier_context": '{"email":"bob@example.com"}',
            },
        )

    assert resp.status_code == 200
    assert captured["user_identifier_context"] == {"email": "bob@example.com"}


@pytest.mark.unit
def test_stream_endpoint_rejects_malformed_json(client):
    """The stream route runs the same parser before constructing the
    generator; malformed JSON must 400 immediately."""
    with patch(
        "api.routers.video_query.stream_video_query"
    ) as mock_stream:
        resp = client.post(
            "/video-query/stream",
            data={
                "query": "x",
                "mode": "graph_state",
                "user_identifier_context": "not-json{{",
            },
        )

    assert resp.status_code == 400
    assert "JSON object" in resp.json()["detail"]
    mock_stream.assert_not_called()


# ---------------------------------------------------------------------------
# Service-layer wiring: acl_callback reaches the pipeline constructor
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_service_wires_acl_callback_into_pipeline(client):
    """The most load-bearing line in video_query_service.py is the one
    that hands get_acl_callback()'s return value to VideoQueryPipeline.
    Patch the pipeline class and assert the kwarg arrives."""

    async def sentinel_callback(video_ids, user_ctx):
        return None  # never invoked in this test

    with patch(
        "api.services.video_query_service.get_acl_callback",
        return_value=sentinel_callback,
    ), patch(
        "api.services.video_query_service.VideoQueryPipeline"
    ) as MockPipeline:
        instance = MockPipeline.return_value
        instance.query = AsyncMock(return_value={"answer": "ok"})
        instance.close = AsyncMock()

        resp = _post_query(
            client,
            user_identifier_context='{"email":"alice@example.com"}',
        )

    assert resp.status_code == 200
    # Constructor kwargs include the callback we wired.
    _, kwargs = MockPipeline.call_args
    assert kwargs["acl_callback"] is sentinel_callback
    # query() received the parsed user_ctx dict.
    instance.query.assert_awaited_once()
    _, query_kwargs = instance.query.call_args
    assert query_kwargs["user_identifier_context"] == {"email": "alice@example.com"}
