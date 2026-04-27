from __future__ import annotations

import json
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from mmct.acl.check_access import AccessCheckResult
from mmct.acl.request_context import (
    UserIdentifierContext,
    get_user_identifier_context,
)

_log = logger.bind(component="acl")

AccessCheckCallback = Callable[
    [list[str], UserIdentifierContext], Awaitable[AccessCheckResult]
]


class ACLFilter:
    """Filters a list of video_ids down to the ones the caller can access.

    The callback contract is intentionally minimal: ``(video_ids, user_ctx)
    -> AccessCheckResult``. Any backend-specific lookups (MS Graph
    drive_id/item_id resolution, custom DB queries, in-memory maps) live
    inside the callback itself — this class does not reach across to them.

    Fail-closed semantics: ``access_denied`` and ``check_failed`` buckets
    are dropped; only ``access_allowed`` survives.
    """

    def __init__(self, callback: AccessCheckCallback) -> None:
        self._callback = callback

    async def filter_video_ids(
        self, video_ids: list[str], user_ctx: UserIdentifierContext
    ) -> set[str]:
        if not video_ids:
            return set()
        result = await self._callback(video_ids, user_ctx)
        if result.access_denied or result.check_failed:
            _log.info(
                "ACL filter dropped videos: denied={} check_failed={}",
                len(result.access_denied),
                len(result.check_failed),
            )
        return set(result.access_allowed)


async def _apply_filter_fail_closed(
    acl_filter: ACLFilter,
    video_ids: list[str],
    user_ctx: UserIdentifierContext,
) -> set[str]:
    """Call ACLFilter.filter_video_ids; on any exception, drop everything.

    Auth/rate-limit/network errors from the callback should not leak raw
    content; converting them to "no videos allowed" preserves fail-closed.
    """
    try:
        return await acl_filter.filter_video_ids(video_ids, user_ctx)
    except Exception as exc:
        _log.warning("ACL filter errored — failing closed (dropping all): {}", exc)
        return set()


def _require_user_ctx() -> UserIdentifierContext:
    """Read the per-request user identifier context or raise.

    The pipeline-level entry point is supposed to enforce that a context
    exists before any tool runs; this is a defense-in-depth check at the
    actual filter call site.
    """
    user_ctx = get_user_identifier_context()
    if user_ctx is None:
        raise ValueError(
            "ACL is enabled but no user_identifier_context is set. "
            "Pipeline.query() must receive a user_identifier_context dict."
        )
    return user_ctx


ExtractSections = Callable[[dict[str, Any]], dict[str, list[Any]]]
PostFilter = Callable[[dict[str, Any]], None]


def _wrap_with_acl(
    tool,
    acl_filter: ACLFilter,
    extract_sections: ExtractSections,
    post_filter: Optional[PostFilter] = None,
):
    """Generic ACL post-filter wrapper for a JSON-returning async tool.

    ``extract_sections`` maps the parsed tool output to a dict of
    ``{section_key: entries}`` whose ``video_id`` fields we filter.
    ``post_filter`` (optional) gets the final dict to recompute derived
    fields like ``total_found``.
    """

    @wraps(tool)
    async def wrapper(*args, **kwargs):
        raw = await tool(*args, **kwargs)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return raw
        if not isinstance(data, dict) or "error" in data:
            return raw

        sections = extract_sections(data)
        video_ids = {
            e["video_id"]
            for entries in sections.values()
            for e in entries
            if isinstance(e, dict) and e.get("video_id")
        }
        if not video_ids:
            return raw

        user_ctx = _require_user_ctx()
        allowed = await _apply_filter_fail_closed(
            acl_filter, list(video_ids), user_ctx
        )
        for key, entries in sections.items():
            data[key] = [
                e for e in entries if isinstance(e, dict) and e.get("video_id") in allowed
            ]
        if post_filter is not None:
            post_filter(data)
        return json.dumps(data)

    return wrapper


# ---------------------------------------------------------------------------
# Per-tool wrappers (graph_agent)
# ---------------------------------------------------------------------------


def _wrap_single_section(
    tool, acl_filter: ACLFilter, key: str, *, recompute_total: bool = False
):
    """Wrapper for tools whose video_id-bearing entries live under one key."""

    def _extract(data: dict[str, Any]) -> dict[str, list[Any]]:
        entries = data.get(key)
        return {key: entries} if isinstance(entries, list) else {}

    post = None
    if recompute_total:
        def post(data: dict[str, Any]) -> None:
            data["total_found"] = len(data.get(key, []))

    return _wrap_with_acl(tool, acl_filter, _extract, post_filter=post)


def wrap_find_relevant_videos(tool, acl_filter: ACLFilter):
    return _wrap_single_section(tool, acl_filter, "relevant_videos", recompute_total=True)


def wrap_traverse_graph(tool, acl_filter: ACLFilter):
    return _wrap_single_section(tool, acl_filter, "results")


def wrap_search_keyframes(tool, acl_filter: ACLFilter):
    return _wrap_single_section(tool, acl_filter, "keyframes")


def wrap_search_graph(tool, acl_filter: ACLFilter):
    """Post-filter search_graph output across every granularity bucket."""

    def _extract(data: dict[str, Any]) -> dict[str, list[Any]]:
        return {k: v for k, v in data.items() if isinstance(v, list)}

    return _wrap_with_acl(tool, acl_filter, _extract)


def wrap_get_video_overview(tool, acl_filter: ACLFilter):
    """Post-filter get_video_overview output.

    Unlike the list-shaped tools, an overview is the content for a single
    video. If that video is denied, replace the body with an
    ``{"error": "access denied for video <id>"}`` shape so the agent
    learns to stop retrying instead of looping over an empty payload.
    """

    @wraps(tool)
    async def wrapper(*args, **kwargs):
        raw = await tool(*args, **kwargs)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return raw
        if not isinstance(data, dict) or "error" in data:
            return raw

        video_id = _extract_overview_video_id(data)
        if video_id is None:
            # Fail-closed: a parseable overview with no extractable video_id is
            # an unexpected shape; we cannot prove the caller has access.
            _log.warning("Overview output had no extractable video_id; failing closed")
            return json.dumps(
                {"error": "access check skipped: video_id missing from overview"}
            )

        user_ctx = _require_user_ctx()
        allowed = await _apply_filter_fail_closed(acl_filter, [video_id], user_ctx)
        if video_id in allowed:
            return raw
        return json.dumps({"error": f"access denied for video {video_id}"})

    return wrapper


def _extract_overview_video_id(data: dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the video_id an overview is describing.

    Tries the top-level ``video_id`` key first; falls back to the first
    node entry that carries one.
    """
    top = data.get("video_id")
    if isinstance(top, str) and top:
        return top
    for value in data.values():
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    vid = entry.get("video_id")
                    if isinstance(vid, str) and vid:
                        return vid
    return None
