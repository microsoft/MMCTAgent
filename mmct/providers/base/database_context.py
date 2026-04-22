"""Per-request graph database name override using ``contextvars``.

This module is **provider-agnostic** — it works with any graph database backend
(Neo4j, Neptune, TigerGraph, etc.).  The three graph-provider base classes
(``BaseGraphQueryProvider``, ``BaseGraphDBProvider``, ``BaseGraphStoreProvider``)
each expose a ``get_database(default)`` method that consults the context
variable defined here before falling back to the provider's instance default.

Concurrency safety
------------------
``ContextVar`` is Python's built-in mechanism for per-task state in asyncio.
Each ``asyncio.Task`` receives an independent copy of the context, so
concurrent requests never interfere with one another.  This is the same
pattern used by Starlette (request state), OpenTelemetry (trace propagation),
and structlog (bound loggers).

Usage
-----
At the **entry point** of a request (MCP tool handler, API route, script):

.. code-block:: python

    from mmct.providers.base.database_context import database_override

    async with database_override("my_other_db"):
        result = await pipeline.query(...)

Inside a **provider implementation**, use the inherited ``get_database``:

.. code-block:: python

    db = self.get_database(self._database)
    async with self._driver.session(database=db) as session:
        ...
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Optional

_graph_database_override: ContextVar[Optional[str]] = ContextVar(
    "_graph_database_override", default=None
)


@asynccontextmanager
async def database_override(name: Optional[str]) -> AsyncIterator[None]:
    """Temporarily override the graph database used by providers in this async context.

    Args:
        name: Database name to use, or ``None`` to keep the provider default.
    """
    if name is None:
        yield
        return
    token = _graph_database_override.set(name)
    try:
        yield
    finally:
        _graph_database_override.reset(token)


def get_database_override() -> Optional[str]:
    """Return the current per-request database override, or ``None``."""
    return _graph_database_override.get(None)
