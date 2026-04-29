"""ASGI middleware that binds the ACL caller-identity dict to a ContextVar.

Reads the ``MMCT-User-Identifier-Context`` HTTP header (a JSON-encoded
freeform object) and enters ``user_identifier_scope`` for the duration
of the downstream call. The dict's shape is a private contract between
the deployer and the configured ACL callback; this middleware does not
introspect any keys.

Strict semantics, mirroring ``api/routers/video_query.py``:

* Missing/empty header → no scope set; request proceeds normally.
* Malformed JSON or non-object value → 400 immediately.

Health/docs paths are skipped so probes don't pay parse cost.
"""
from __future__ import annotations

import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from mmct.acl.request_context import user_identifier_scope

ACL_HEADER = "MMCT-User-Identifier-Context"

# Routes that don't dispatch MCP tools — middleware skips parse work for them.
_ACL_SKIP_PATHS: frozenset[str] = frozenset({"/", "/readyz", "/api", "/docs"})


class ACLContextMiddleware(BaseHTTPMiddleware):
    """Decode the caller-identity header and scope it for the request."""

    async def dispatch(self, request, call_next):
        if request.url.path in _ACL_SKIP_PATHS:
            return await call_next(request)

        raw = request.headers.get(ACL_HEADER)
        if not raw:
            return await call_next(request)

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            return JSONResponse(
                {"detail": f"{ACL_HEADER} must be a JSON object ({exc.msg})"},
                status_code=400,
            )
        if not isinstance(decoded, dict):
            return JSONResponse(
                {
                    "detail": f"{ACL_HEADER} must be a JSON object, not a "
                              f"{type(decoded).__name__}"
                },
                status_code=400,
            )

        async with user_identifier_scope(decoded):
            return await call_next(request)
