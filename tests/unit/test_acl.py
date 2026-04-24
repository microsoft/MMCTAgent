from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from loguru import logger

from mmct.acl import (
    GraphAPIError,
    GraphAuthenticationError,
    GraphRateLimitError,
    VideoIdentifier,
    check_access_to_video,
    check_access_to_video_list,
)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN = "test-token-value"
DRIVE_ID = "drive123"
ITEM_ID = "item456"
VIDEO_ID = "vid_abc"


def _item_url(drive_id: str = DRIVE_ID, item_id: str = ITEM_ID) -> str:
    return f"{GRAPH_BASE}/drives/{drive_id}/items/{item_id}"


def _make_vid(
    video_id: str = VIDEO_ID, drive_id: str = DRIVE_ID, item_id: str = ITEM_ID
) -> VideoIdentifier:
    return VideoIdentifier(video_id=video_id, drive_id=drive_id, item_id=item_id)


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_granted_200_with_matching_id():
    respx.get(_item_url()).mock(return_value=httpx.Response(200, json={"id": ITEM_ID}))
    async with httpx.AsyncClient() as client:
        result = await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)
    assert result is True


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_denied_403():
    respx.get(_item_url()).mock(return_value=httpx.Response(403))
    async with httpx.AsyncClient() as client:
        result = await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)
    assert result is False


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_denied_404():
    respx.get(_item_url()).mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        result = await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)
    assert result is False


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_auth_error_401():
    respx.get(_item_url()).mock(return_value=httpx.Response(401))
    async with httpx.AsyncClient() as client:
        with pytest.raises(GraphAuthenticationError):
            await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_rate_limit_429():
    respx.get(_item_url()).mock(return_value=httpx.Response(429))
    async with httpx.AsyncClient() as client:
        with pytest.raises(GraphRateLimitError):
            await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_server_error_500():
    respx.get(_item_url()).mock(return_value=httpx.Response(500, text="internal error"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(GraphAPIError) as exc_info:
            await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_payload_mismatch_on_200():
    respx.get(_item_url()).mock(
        return_value=httpx.Response(200, json={"id": "different-item-id"})
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(GraphAPIError) as exc_info:
            await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)
    assert exc_info.value.status_code == 200
    assert "mismatch" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_token_not_logged():
    # Loguru doesn't propagate to stdlib logging; attach a direct sink so we
    # actually see what would have been emitted.
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="DEBUG")
    try:
        respx.get(_item_url()).mock(return_value=httpx.Response(200, json={"id": ITEM_ID}))
        async with httpx.AsyncClient() as client:
            await check_access_to_video(client, TOKEN, DRIVE_ID, ITEM_ID)
    finally:
        logger.remove(sink_id)

    assert captured, "expected at least one log message"
    for msg in captured:
        assert TOKEN not in msg


@pytest.mark.asyncio
@pytest.mark.unit
async def test_empty_token_raises_auth_error():
    async with httpx.AsyncClient() as client:
        with pytest.raises(GraphAuthenticationError):
            await check_access_to_video(client, "", DRIVE_ID, ITEM_ID)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_empty_list_returns_empty_result():
    result = await check_access_to_video_list(TOKEN, [])
    assert result.access_allowed == []
    assert result.access_denied == []
    assert result.check_failed == []


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_mixed_results():
    vid1 = _make_vid("v1", "d1", "i1")
    vid2 = _make_vid("v2", "d2", "i2")
    vid3 = _make_vid("v3", "d3", "i3")

    respx.get(_item_url("d1", "i1")).mock(
        return_value=httpx.Response(200, json={"id": "i1"})
    )
    respx.get(_item_url("d2", "i2")).mock(return_value=httpx.Response(200, json={"id": "i2"}))
    respx.get(_item_url("d3", "i3")).mock(return_value=httpx.Response(404))

    result = await check_access_to_video_list(TOKEN, [vid1, vid2, vid3])

    assert sorted(result.access_allowed) == ["v1", "v2"]
    assert result.access_denied == ["v3"]
    assert result.check_failed == []


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_auth_error_propagates_from_batch():
    vid1 = _make_vid("v1", "d1", "i1")
    vid2 = _make_vid("v2", "d2", "i2")

    respx.get(_item_url("d1", "i1")).mock(return_value=httpx.Response(200, json={"id": "i1"}))
    respx.get(_item_url("d2", "i2")).mock(return_value=httpx.Response(401))

    with pytest.raises(GraphAuthenticationError):
        await check_access_to_video_list(TOKEN, [vid1, vid2])


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_graph_api_error_treated_as_check_failed():
    vid1 = _make_vid("v1", "d1", "i1")
    vid2 = _make_vid("v2", "d2", "i2")

    respx.get(_item_url("d1", "i1")).mock(return_value=httpx.Response(200, json={"id": "i1"}))
    respx.get(_item_url("d2", "i2")).mock(return_value=httpx.Response(500, text="oops"))

    result = await check_access_to_video_list(TOKEN, [vid1, vid2])

    assert result.access_allowed == ["v1"]
    assert result.access_denied == []
    assert result.check_failed == ["v2"]


@pytest.mark.asyncio
@pytest.mark.unit
@respx.mock
async def test_concurrency_limit_respected():
    n = 8
    max_concurrency = 3
    concurrent_count = 0
    peak_concurrent = 0

    vids = [_make_vid(f"v{i}", f"d{i}", f"i{i}") for i in range(n)]

    async def slow_response(request: httpx.Request) -> httpx.Response:
        nonlocal concurrent_count, peak_concurrent
        concurrent_count += 1
        peak_concurrent = max(peak_concurrent, concurrent_count)
        await asyncio.sleep(0.01)
        concurrent_count -= 1
        item_id = request.url.path.split("/")[-1]
        return httpx.Response(200, json={"id": item_id})

    for vid in vids:
        respx.get(_item_url(vid.drive_id, vid.item_id)).mock(side_effect=slow_response)

    await check_access_to_video_list(TOKEN, vids, max_concurrency=max_concurrency)

    assert peak_concurrent <= max_concurrency
