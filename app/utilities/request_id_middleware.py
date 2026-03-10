"""Request ID Middleware - Generates and propagates a unique request ID per request.

Stores the request_id in a ContextVar so all downstream code (including
async generators for SSE streaming) can access it via get_request_id().
"""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ContextVar holds the current request's ID
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Return the request ID for the current context."""
    return _request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a UUID4 request_id to every incoming request.

    - If the client sends an ``X-Request-ID`` header, that value is used.
    - Otherwise a new UUID4 is generated.
    - The request_id is stored in a ContextVar and set as an
      ``X-Request-ID`` response header.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use client-provided ID or generate a new one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in context so downstream code can access it
        token = _request_id_ctx.set(request_id)

        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            _request_id_ctx.reset(token)
