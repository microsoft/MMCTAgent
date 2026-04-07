"""Request ID middleware for MMCT Agent.

Assigns a unique request ID to every incoming HTTP request and propagates it
through the async call stack via a ContextVar, making it available to service
and agent code without explicit parameter threading.

Header conventions:
  - Inbound  : ``X-MMCT-Request-ID`` — client-supplied ID (optional)
  - Outbound : ``X-MMCT-Request-ID`` — ID used for this request (echoed or generated)

If the client does not supply ``X-MMCT-Request-ID``, a new UUID4 is generated.
The legacy ``X-Request-ID`` header is also checked as a fallback so that
existing clients are not broken during a transition period.
"""

import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Primary header name for MMCT request correlation
MMCT_REQUEST_ID_HEADER = "X-MMCT-Request-ID"
# Legacy fallback — accepted on inbound requests only
_LEGACY_HEADER = "X-Request-ID"

# ContextVar holds the current request's correlation ID
_request_id_ctx: ContextVar[str] = ContextVar("mmct_request_id", default="")


def get_request_id() -> str:
    """Return the MMCT request ID for the current async context.

    Returns an empty string when called outside a request context (e.g. in
    background tasks or tests that do not exercise the middleware).
    """
    return _request_id_ctx.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a correlation ID to every incoming request.

    Resolution order for the request ID:
      1. ``X-MMCT-Request-ID`` header from the client
      2. ``X-Request-ID`` header (legacy fallback)
      3. Freshly generated UUID4

    The resolved ID is:
      - Stored in a ContextVar so all downstream async code can read it via
        ``get_request_id()`` without needing it passed as a parameter.
      - Echoed back in the ``X-MMCT-Request-ID`` response header for log
        correlation on the client side.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = (
            request.headers.get(MMCT_REQUEST_ID_HEADER)
            or request.headers.get(_LEGACY_HEADER)
            or str(uuid.uuid4())
        )

        token = _request_id_ctx.set(request_id)
        try:
            response: Response = await call_next(request)
            response.headers[MMCT_REQUEST_ID_HEADER] = request_id
            return response
        finally:
            _request_id_ctx.reset(token)
