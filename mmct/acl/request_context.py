"""Per-request user identifier context for ACL access checks.

Mirrors the shape of ``mmct/providers/base/database_context.py``: a
``ContextVar`` carries a free-form ``dict`` of caller-supplied identity data
(user email, MS Graph token, custom keys, …) for the duration of a single
asyncio task. Callers open the scope at request entry; ACL filter wrappers
read the dict at filter time and forward it to the access-check callback.

The dict's shape is a free contract between the caller and whichever
``acl_callback`` was wired into ``VideoQueryPipeline``. The ACL module
itself does not introspect any keys.

.. warning::
    The dict commonly carries bearer secrets (e.g. ``graph_token``).
    **Never** log it, ``repr`` it, format it into exception messages, or
    return it in API responses. Loguru's default backtrace + diagnose
    formatter prints frame locals; if a `user_identifier_context` is in
    scope on an exception frame, its contents (including any token)
    will be written to logs. Callers are responsible for redaction
    discipline at every layer they own.

Concurrency
-----------
Each ``asyncio.Task`` receives an independent copy of the context, so
concurrent requests cannot leak identity data into one another. This is
the same primitive used by Starlette (request state), OpenTelemetry
(trace propagation), and this repo's ``database_override``.

Usage
-----
At a request entry point (e.g. ``VideoQueryPipeline.query``)::

    from mmct.acl.request_context import user_identifier_scope

    async with user_identifier_scope({"graph_token": token, "email": addr}):
        await orchestrator.query(...)

Inside an ACL wrapper or filter call site::

    from mmct.acl.request_context import get_user_identifier_context

    user_ctx = get_user_identifier_context()
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, AsyncIterator, Optional

UserIdentifierContext = dict[str, Any]

_user_identifier_context_var: ContextVar[Optional[UserIdentifierContext]] = ContextVar(
    "_user_identifier_context", default=None
)


@asynccontextmanager
async def user_identifier_scope(
    ctx: Optional[UserIdentifierContext],
) -> AsyncIterator[None]:
    """Bind ``ctx`` to the current asyncio task for the duration of the block.

    Passing ``None`` is a no-op — the var stays at whatever it inherited.
    """
    if ctx is None:
        yield
        return
    token = _user_identifier_context_var.set(ctx)
    try:
        yield
    finally:
        _user_identifier_context_var.reset(token)


def get_user_identifier_context() -> Optional[UserIdentifierContext]:
    """Return the current per-request identifier context, or ``None``."""
    return _user_identifier_context_var.get()
