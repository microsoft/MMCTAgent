"""Example: Auth state hook for the graph_state pipeline.

Demonstrates how to intercept state transitions to enforce
authorization — filtering ``effective_video_ids`` after discovery
and stripping unauthorized evidence after retrieval.

Usage::

    from scripts.custom_steps.auth_state_hook_example import AuthStateHook

    hook = AuthStateHook(auth_service=my_auth_service)
    orchestrator = StateOrchestrator(
        model_client=client,
        neo4j_provider=provider,
        state_hooks=[hook],
    )
    result = await orchestrator.query(
        "What topics are covered?",
        query_context={"user_id": "user_123"},
    )
"""

from typing import Any, Dict, List, Optional, Protocol, Set

from loguru import logger

from mmct.video_pipeline.graph_state.hooks import StateHook
from mmct.video_pipeline.graph_state.state_machine import QueryContext, QueryState
from mmct.video_pipeline.graph_agent.middleware import get_query_context

_log = logger.bind(component="auth_state_hook")


class AuthService(Protocol):
    """Protocol for an authorization service."""

    async def get_allowed_video_ids(self, user_id: str) -> List[str]:
        """Return video IDs the user is authorized to access."""
        ...


class AuthStateHook(StateHook):
    """Filters video IDs and evidence to authorized videos only.

    Hooks into:
    - **DISCOVER_VIDEOS** (after): constrains ``effective_video_ids`` to
      only those the user can access.
    - **RETRIEVE** (after): strips evidence from unauthorized videos.
    - **PARSE_INPUT** (after): for single/multi-video scope, constrains
      ``effective_video_ids`` upfront.

    Per-query caching ensures the auth service is called at most once
    per query execution.
    """

    def __init__(self, auth_service: AuthService):
        self._auth_service = auth_service
        self._cache: Dict[str, Set[str]] = {}

    def applies_to(self, state: QueryState) -> bool:
        return state in {
            QueryState.PARSE_INPUT,
            QueryState.DISCOVER_VIDEOS,
            QueryState.RETRIEVE,
        }

    async def _get_allowed(self, user_id: str) -> Set[str]:
        """Fetch allowed video IDs (cached per user_id within a query)."""
        if user_id not in self._cache:
            allowed = await self._auth_service.get_allowed_video_ids(user_id)
            self._cache[user_id] = set(allowed)
        return self._cache[user_id]

    async def after_state(
        self, state: QueryState, ctx: QueryContext, next_state: QueryState
    ) -> Optional[QueryState]:
        qctx = get_query_context()
        user_id = qctx.get("user_id")
        if not user_id:
            return None

        allowed = await self._get_allowed(user_id)
        if not allowed:
            _log.warning(f"User {user_id} has no allowed videos — routing to ERROR")
            ctx.answer = "You do not have access to any videos."
            return QueryState.ERROR

        if state == QueryState.PARSE_INPUT:
            if ctx.effective_video_ids:
                before = len(ctx.effective_video_ids)
                ctx.effective_video_ids = [
                    v for v in ctx.effective_video_ids if v in allowed
                ]
                _log.info(
                    f"Auth: filtered effective_video_ids "
                    f"{before} → {len(ctx.effective_video_ids)}"
                )

        elif state == QueryState.DISCOVER_VIDEOS:
            if ctx.effective_video_ids:
                before = len(ctx.effective_video_ids)
                ctx.effective_video_ids = [
                    v for v in ctx.effective_video_ids if v in allowed
                ]
                _log.info(
                    f"Auth: filtered discovered videos "
                    f"{before} → {len(ctx.effective_video_ids)}"
                )
                if not ctx.effective_video_ids:
                    ctx.answer = "No authorized videos found for your query."
                    return QueryState.SUBMIT

        elif state == QueryState.RETRIEVE:
            before = len(ctx.evidence)
            ctx.evidence = [
                e for e in ctx.evidence if e.get("video_id") in allowed
            ]
            _log.info(
                f"Auth: filtered evidence {before} → {len(ctx.evidence)}"
            )
            if ctx.keyframes:
                before_kf = len(ctx.keyframes)
                ctx.keyframes = [
                    kf for kf in ctx.keyframes if kf.get("video_id") in allowed
                ]
                _log.info(
                    f"Auth: filtered keyframes {before_kf} → {len(ctx.keyframes)}"
                )

        return None

    def clear_cache(self) -> None:
        """Clear the per-query auth cache. Call between queries."""
        self._cache.clear()
