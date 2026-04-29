"""Tests for the MCP server's ACL caller-identity HTTP middleware.

Covers ``ACLContextMiddleware`` in ``mcp_server/asgi.py``: header parse
semantics and ContextVar propagation to the downstream ASGI app. Tests
exercise the middleware in isolation against a tiny probe app rather
than booting the full FastMCP app, since the middleware is the unit
under test (and the MCP app pulls in heavy ML deps).
"""
from __future__ import annotations

import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp_server.acl_middleware import ACL_HEADER, ACLContextMiddleware
from mmct.acl.request_context import get_user_identifier_context


def _build_probe_app() -> Starlette:
    """Starlette app with the ACL middleware and a probe route that
    returns whatever ``get_user_identifier_context()`` sees downstream."""

    async def probe(request):
        return JSONResponse({"ctx": get_user_identifier_context()})

    app = Starlette(routes=[Route("/probe", probe)])
    app.add_middleware(ACLContextMiddleware)
    return app


@pytest.fixture
def probe_app():
    return _build_probe_app()


async def _get(app, path: str, headers: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        return await c.get(path, headers=headers or {})


# ---------------------------------------------------------------------------
# Valid header → ContextVar populated downstream
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_header_sets_context_var(probe_app):
    payload = {"email": "alice@example.com", "graph_token": "abc"}
    resp = await _get(probe_app, "/probe", {ACL_HEADER: json.dumps(payload)})
    assert resp.status_code == 200
    assert resp.json()["ctx"] == payload


# ---------------------------------------------------------------------------
# Missing/empty header → ContextVar stays None, request proceeds
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_header_passes_through(probe_app):
    resp = await _get(probe_app, "/probe")
    assert resp.status_code == 200
    assert resp.json()["ctx"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_header_passes_through(probe_app):
    resp = await _get(probe_app, "/probe", {ACL_HEADER: ""})
    assert resp.status_code == 200
    assert resp.json()["ctx"] is None


# ---------------------------------------------------------------------------
# Malformed/non-object header → 400 (strict, mirrors FastAPI route)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    ["not-json{{", '"alice"', "[1,2,3]", "42", "null"],
    ids=["malformed", "string", "array", "number", "null"],
)
async def test_invalid_header_returns_400(probe_app, payload):
    resp = await _get(probe_app, "/probe", {ACL_HEADER: payload})
    assert resp.status_code == 400
    assert "JSON object" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Skip-list paths bypass parse logic entirely
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_skip_list_path_bypasses_parse():
    """A skip-listed path should not 400 even on malformed header,
    because the middleware short-circuits before parsing."""

    async def root(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/", root), Route("/readyz", root)])
    app.add_middleware(ACLContextMiddleware)

    for path in ("/", "/readyz"):
        resp = await _get(app, path, {ACL_HEADER: "not-json{{"})
        assert resp.status_code == 200, f"{path} should bypass middleware"
        assert resp.json() == {"ok": True}
