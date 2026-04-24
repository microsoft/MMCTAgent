from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx
from loguru import logger

from mmct.utils.error_handler import ProviderException

_log = logger.bind(component="acl")

GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"


class GraphACLError(ProviderException):
    """Base class for MS Graph ACL errors."""


class GraphAuthenticationError(GraphACLError):
    """HTTP 401 — token missing, invalid, or expired. Fails the whole batch."""


class GraphRateLimitError(GraphACLError):
    """HTTP 429 — MS Graph rate limit hit. Fails the whole batch."""


class GraphAPIError(GraphACLError):
    """HTTP 5xx or unexpected status — treated as check_failed in batches."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Graph API error {status_code}: {detail}")
        self.status_code = status_code


@dataclass(frozen=True)
class VideoIdentifier:
    video_id: str
    drive_id: str
    item_id: str


@dataclass
class AccessCheckResult:
    access_allowed: list[str] = field(default_factory=list)
    access_denied: list[str] = field(default_factory=list)
    check_failed: list[str] = field(default_factory=list)


@dataclass
class _SingleResult:
    video_id: str
    granted: bool | None


async def check_access_to_video(
    client: httpx.AsyncClient,
    graph_token: str,
    drive_id: str,
    item_id: str,
) -> bool:
    """Return True if the token bearer has read access to the given OneDrive item.

    Raises GraphAuthenticationError on 401, GraphRateLimitError on 429,
    and GraphAPIError on other unexpected statuses or malformed 200 payloads.
    Callers configure timeouts on the passed httpx client.
    """
    if not graph_token:
        raise GraphAuthenticationError("graph_token is empty or None")

    url = (
        f"{GRAPH_API_BASE}/drives/{quote(drive_id, safe='')}"
        f"/items/{quote(item_id, safe='')}"
    )
    headers = {"Authorization": f"Bearer {graph_token}"}

    _log.debug("Checking Graph access drive_id={} item_id={}", drive_id, item_id)

    response = await client.get(url, headers=headers)
    status = response.status_code

    if status == 200:
        body = response.json()
        if not isinstance(body, dict) or body.get("id") != item_id:
            raise GraphAPIError(200, f"item_id mismatch: expected {item_id}")
        _log.debug("Access granted drive_id={} item_id={}", drive_id, item_id)
        return True

    if status in (403, 404):
        _log.debug("Access denied ({}) drive_id={} item_id={}", status, drive_id, item_id)
        return False

    if status == 401:
        _log.warning("Graph authentication failure drive_id={} item_id={}", drive_id, item_id)
        raise GraphAuthenticationError("MS Graph returned 401 — token invalid or expired")

    if status == 429:
        _log.warning("Graph rate limit hit drive_id={} item_id={}", drive_id, item_id)
        raise GraphRateLimitError("MS Graph returned 429 — rate limited")

    # Never log or include response.text — upstream proxies could echo Bearer tokens.
    _log.warning(
        "Unexpected Graph status {} drive_id={} item_id={}", status, drive_id, item_id
    )
    raise GraphAPIError(status, f"unexpected status {status}")


async def check_access_to_video_list(
    graph_token: str,
    video_identifiers: list[VideoIdentifier],
    *,
    max_concurrency: int = 10,
) -> AccessCheckResult:
    """Check MS Graph access for a batch of videos, returning a three-bucket result.

    GraphAuthenticationError and GraphRateLimitError on any item abort the whole
    batch (propagated to the caller). Other errors (5xx, network, malformed
    payload) bucket the affected video into `check_failed` so the caller can
    fail closed without losing observability.
    """
    if not video_identifiers:
        return AccessCheckResult()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _check_one(client: httpx.AsyncClient, vid: VideoIdentifier) -> _SingleResult:
        async with semaphore:
            try:
                granted = await check_access_to_video(
                    client, graph_token, vid.drive_id, vid.item_id
                )
                return _SingleResult(video_id=vid.video_id, granted=granted)
            except (GraphAuthenticationError, GraphRateLimitError):
                raise
            except (GraphAPIError, httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                _log.warning(
                    "Graph API error for drive_id={} item_id={}: {}",
                    vid.drive_id,
                    vid.item_id,
                    exc,
                )
                return _SingleResult(video_id=vid.video_id, granted=None)

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [_check_one(client, vid) for vid in video_identifiers]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    result = AccessCheckResult()
    for raw in raw_results:
        if isinstance(raw, Exception):
            raise raw
        if raw.granted is True:
            result.access_allowed.append(raw.video_id)
        elif raw.granted is False:
            result.access_denied.append(raw.video_id)
        else:
            result.check_failed.append(raw.video_id)

    return result
