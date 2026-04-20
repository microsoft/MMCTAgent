"""Example: Auth middleware for the graph_agent pipeline.

Demonstrates how to use ``ToolMiddleware`` to enforce per-user video
access controls on all graph retrieval tools — without modifying
core library code and with zero extra LLM calls.

The middleware:
- **before_tool_call**: restricts ``video_ids`` / ``video_id`` arguments
  to only videos the user is authorized to access.
- **after_tool_call**: strips any unauthorized video data that may have
  leaked through tools that don't accept a ``video_ids`` filter.

Usage::

    import scripts.custom_steps.auth_middleware_example as auth_mod

    auth_mw = auth_mod.AuthMiddleware(auth_service=my_auth_service)

    orchestrator = GraphOrchestrator(
        model_client=client,
        neo4j_provider=provider,
        tool_middleware=[auth_mw],
    )

    result = await orchestrator.query(
        "What topics are covered?",
        query_context={"user_id": "user_123"},
    )
"""

import json
from typing import Any, Dict, List, Optional, Protocol

from loguru import logger

from mmct.video_pipeline.graph_agent.middleware import (
    ToolMiddleware,
    get_query_context,
)

_log = logger.bind(component="AuthMiddleware")


# ---------------------------------------------------------------------------
# Auth service protocol — implement this for your backend
# ---------------------------------------------------------------------------


class AuthService(Protocol):
    """Protocol for an authorization service that resolves allowed videos."""

    async def get_allowed_video_ids(self, user_id: str) -> List[str]:
        """Return video IDs the user is permitted to access."""
        ...


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

# Tools that accept a video_ids (list) parameter
_TOOLS_WITH_VIDEO_IDS = {
    "search_graph",
    "find_relevant_videos",
    "search_keyframes",
}

# Tools that accept a video_id (single) parameter
_TOOLS_WITH_VIDEO_ID = {
    "get_video_overview",
    "traverse_graph",
}


class AuthMiddleware(ToolMiddleware):
    """Restricts tool calls to only authorized videos.

    Reads ``user_id`` from the per-query context (set via
    ``query_context={"user_id": "..."}`` on the orchestrator) and
    queries the auth service for the allowed video list.

    Args:
        auth_service: An object implementing ``AuthService``.
        cache_per_query: If True (default), caches the allowed list
            for the duration of a single query to avoid repeated
            auth service calls.
    """

    def __init__(self, auth_service: AuthService, cache_per_query: bool = True):
        self.auth_service = auth_service
        self.cache_per_query = cache_per_query
        self._cached_allowed: Optional[List[str]] = None
        self._cached_user_id: Optional[str] = None

    def applies_to(self, tool_name: str) -> bool:
        return tool_name in (_TOOLS_WITH_VIDEO_IDS | _TOOLS_WITH_VIDEO_ID)

    async def _get_allowed(self, user_id: str) -> List[str]:
        """Resolve allowed videos, with optional per-query caching."""
        if (
            self.cache_per_query
            and self._cached_user_id == user_id
            and self._cached_allowed is not None
        ):
            return self._cached_allowed

        allowed = await self.auth_service.get_allowed_video_ids(user_id)
        if self.cache_per_query:
            self._cached_user_id = user_id
            self._cached_allowed = allowed
        return allowed

    async def before_tool_call(
        self, tool_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = get_query_context()
        user_id = ctx.get("user_id")
        if not user_id:
            _log.warning(f"No user_id in query_context — skipping auth for {tool_name}")
            return kwargs

        allowed = await self._get_allowed(user_id)
        allowed_set = set(allowed)

        if tool_name in _TOOLS_WITH_VIDEO_IDS:
            current = kwargs.get("video_ids")
            if current is not None:
                kwargs["video_ids"] = [v for v in current if v in allowed_set]
            else:
                kwargs["video_ids"] = allowed
            _log.debug(
                f"{tool_name}: filtered video_ids to {len(kwargs['video_ids'])} allowed"
            )

        elif tool_name in _TOOLS_WITH_VIDEO_ID:
            current = kwargs.get("video_id")
            if current and current not in allowed_set:
                _log.warning(
                    f"{tool_name}: blocked access to unauthorized video '{current}'"
                )
                kwargs["video_id"] = ""

        return kwargs

    async def after_tool_call(
        self, tool_name: str, result: str, kwargs: Dict[str, Any]
    ) -> str:
        """Strip results referencing unauthorized videos."""
        ctx = get_query_context()
        user_id = ctx.get("user_id")
        if not user_id:
            return result

        allowed_set = set(await self._get_allowed(user_id))

        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result

        filtered = self._filter_response(data, allowed_set)
        return json.dumps(filtered)

    def _filter_response(
        self, data: Any, allowed: set
    ) -> Any:
        """Recursively remove entries with unauthorized video_ids."""
        if isinstance(data, dict):
            # Filter list values that contain video_id entries
            filtered = {}
            for key, value in data.items():
                if isinstance(value, list):
                    filtered[key] = [
                        item
                        for item in value
                        if not (
                            isinstance(item, dict)
                            and "video_id" in item
                            and item["video_id"] not in allowed
                        )
                    ]
                else:
                    filtered[key] = value

            # Update counts if present
            for count_key in ("total", "total_found"):
                if count_key in filtered:
                    for list_key, list_val in filtered.items():
                        if isinstance(list_val, list) and list_key != count_key:
                            filtered[count_key] = len(list_val)
                            break

            return filtered
        return data

    def reset_cache(self) -> None:
        """Clear the per-query cache.  Called automatically between queries."""
        self._cached_allowed = None
        self._cached_user_id = None
