from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from mmct.acl.check_access import (
    AccessCheckResult,
    VideoIdentifier,
    check_access_to_video_list,
)

_log = logger.bind(component="acl")

AccessCheckCallback = Callable[[list[str]], Awaitable[AccessCheckResult]]
VideoIdentifierLookup = Callable[[list[str]], Awaitable[list[VideoIdentifier]]]


@dataclass(frozen=True)
class ACLContext:
    """Configuration bundle for ACL filtering.

    Valid configurations:
      - `access_check_callback` set (custom backend). `graph_token` and
        `video_identifier_lookup` are ignored.
      - `access_check_callback` unset AND both `graph_token` and
        `video_identifier_lookup` set (default MS Graph adapter).

    Any other combination raises ValueError at construction time.

    `graph_token` is excluded from the auto-generated repr so an accidental
    log of the context doesn't leak the bearer token.
    """

    access_check_callback: Optional[AccessCheckCallback] = None
    graph_token: Optional[str] = field(default=None, repr=False)
    video_identifier_lookup: Optional[VideoIdentifierLookup] = None

    def __post_init__(self) -> None:
        if self.access_check_callback is not None:
            return
        if self.graph_token is not None and self.graph_token != "" and self.video_identifier_lookup is not None:
            return
        raise ValueError(
            "ACLContext requires either access_check_callback, or both "
            "graph_token (non-empty) and video_identifier_lookup"
        )


class ACLFilter:
    """Filters a list of video_ids down to the ones the caller can access.

    Fail-closed: access_denied and check_failed buckets are dropped; only
    access_allowed survives. Exceptions raised by the callback propagate to
    the caller, which is responsible for treating them as fail-closed.
    """

    def __init__(self, ctx: ACLContext) -> None:
        self._ctx = ctx
        if ctx.access_check_callback is not None:
            self._callback: AccessCheckCallback = ctx.access_check_callback
        else:
            self._callback = self._build_default_callback(ctx)

    @staticmethod
    def _build_default_callback(ctx: ACLContext) -> AccessCheckCallback:
        graph_token = ctx.graph_token
        lookup = ctx.video_identifier_lookup
        assert graph_token and lookup is not None  # guaranteed by __post_init__

        async def _default_callback(video_ids: list[str]) -> AccessCheckResult:
            identifiers = await lookup(video_ids)
            return await check_access_to_video_list(graph_token, identifiers)

        return _default_callback

    async def filter_video_ids(self, video_ids: list[str]) -> set[str]:
        if not video_ids:
            return set()
        result = await self._callback(video_ids)
        if result.access_denied or result.check_failed:
            _log.info(
                "ACL filter dropped videos: denied={} check_failed={}",
                len(result.access_denied),
                len(result.check_failed),
            )
        return set(result.access_allowed)


async def _apply_filter_fail_closed(
    acl_filter: ACLFilter, video_ids: list[str]
) -> set[str]:
    """Call filter_video_ids; on any exception, treat as fail-closed (drop all).

    The callback contract allows GraphAuthenticationError / GraphRateLimitError
    to propagate; for discovery post-filters we prefer to surface that as an
    empty allowed set rather than letting the exception abort the whole agent
    turn with raw content potentially half-leaked.
    """
    try:
        return await acl_filter.filter_video_ids(video_ids)
    except Exception as exc:
        _log.warning("ACL filter errored — failing closed (dropping all): {}", exc)
        return set()


ExtractSections = Callable[[dict[str, Any]], dict[str, list[Any]]]
PostFilter = Callable[[dict[str, Any]], None]


def _wrap_with_acl(
    tool,
    acl_filter: ACLFilter,
    extract_sections: ExtractSections,
    post_filter: Optional[PostFilter] = None,
):
    """Generic ACL post-filter wrapper for a JSON-returning async tool.

    `extract_sections` maps the parsed tool output to a dict of
    `{section_key: entries}` whose `video_id` fields we filter. After
    filtering, entries are written back in place on `data`. `post_filter`
    gets the final dict to recompute derived fields (e.g., total counts).
    """

    @wraps(tool)
    async def wrapper(*args, **kwargs):
        raw = await tool(*args, **kwargs)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return raw
        # Non-dict payload or error shape → no video entries to filter.
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

        allowed = await _apply_filter_fail_closed(acl_filter, list(video_ids))
        for key, entries in sections.items():
            data[key] = [
                e for e in entries if isinstance(e, dict) and e.get("video_id") in allowed
            ]
        if post_filter is not None:
            post_filter(data)
        return json.dumps(data)

    return wrapper


def wrap_find_relevant_videos(tool, acl_filter: ACLFilter):
    """Post-filter find_relevant_videos output by ACL; recomputes total_found."""

    def _extract(data: dict[str, Any]) -> dict[str, list[Any]]:
        entries = data.get("relevant_videos")
        return {"relevant_videos": entries} if isinstance(entries, list) else {}

    def _recompute_total(data: dict[str, Any]) -> None:
        data["total_found"] = len(data.get("relevant_videos", []))

    return _wrap_with_acl(tool, acl_filter, _extract, post_filter=_recompute_total)


def wrap_search_graph(tool, acl_filter: ACLFilter):
    """Post-filter search_graph output by ACL across every granularity bucket."""

    def _extract(data: dict[str, Any]) -> dict[str, list[Any]]:
        return {k: v for k, v in data.items() if isinstance(v, list)}

    return _wrap_with_acl(tool, acl_filter, _extract)
