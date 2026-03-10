#!/usr/bin/env python3
"""API integration tests for the MMCT FastAPI Docker image.

Tests ALL exposed endpoints against a running container.
Run a container first, then execute:

    # Start container
    docker run -d --name mmct-test -p 8000:8000 \
        --env-file app/.env.gpt4.1 mmct-lively-fastapi:latest

    # Run tests (from repo root)
    python app/test_docker_api.py
    python app/test_docker_api.py --base-url http://localhost:8113
"""

import argparse
import json
import sys
import time
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:8000"
# A known-good video ID present in the Neo4j graph
SAMPLE_VIDEO_ID = "-b5yWSQ_9Sw"
TIMEOUT = 30  # seconds per request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0
errors: list[str] = []


def result(name: str, ok: bool, detail: str = ""):
    global passed, failed
    icon = "✅" if ok else "❌"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {name}{suffix}")
    if ok:
        passed += 1
    else:
        failed += 1
        errors.append(f"{name}: {detail}")


def get(path: str, **kwargs) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT, **kwargs)


def post(path: str, json_body: dict, **kwargs) -> requests.Response:
    return requests.post(
        f"{BASE_URL}{path}", json=json_body, timeout=TIMEOUT, **kwargs
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_root():
    """GET / — root endpoint with version and build info."""
    print("\n[1/10] GET /")
    try:
        r = get("/")
        data = r.json()
        result("status 200", r.status_code == 200, f"got {r.status_code}")
        result("has version", "version" in data, data.get("version", ""))
        result("has build_timestamp", "build_timestamp" in data, data.get("build_timestamp", ""))
        result("has docs_url", data.get("docs_url") == "/docs")
    except Exception as e:
        result("request failed", False, str(e))


def test_health():
    """GET /health — health check."""
    print("\n[2/10] GET /health")
    try:
        r = get("/health")
        data = r.json()
        result("status 200", r.status_code == 200, f"got {r.status_code}")
        result("status healthy", data.get("status") == "healthy")
    except Exception as e:
        result("request failed", False, str(e))


def test_providers():
    """GET /providers — active and supported providers."""
    print("\n[3/10] GET /providers")
    try:
        r = get("/providers")
        data = r.json()
        result("status 200", r.status_code == 200, f"got {r.status_code}")
        result("has active_providers", "active_providers" in data)
        active = data.get("active_providers", {})
        for key in ("llm", "text_embedding", "image_embedding", "graph_query", "storage", "vector_search"):
            result(f"  active_providers.{key}", key in active, active.get(key, {}).get("provider", "missing"))
        result("has all_supported_providers", "all_supported_providers" in data)
    except Exception as e:
        result("request failed", False, str(e))


def test_openapi():
    """GET /openapi.json — Swagger schema contains version and build info."""
    print("\n[4/10] GET /openapi.json")
    try:
        r = get("/openapi.json")
        data = r.json()
        info = data.get("info", {})
        result("status 200", r.status_code == 200, f"got {r.status_code}")
        result("has version", "version" in info, info.get("version", ""))
        result("description has build time", "Built:" in info.get("description", ""))
        paths = list(data.get("paths", {}).keys())
        result(f"paths count >= 10", len(paths) >= 10, f"found {len(paths)}: {paths}")
    except Exception as e:
        result("request failed", False, str(e))


def test_videos():
    """GET /videos — list ingested video IDs."""
    print("\n[5/10] GET /videos")
    try:
        r = get("/videos")
        data = r.json()
        result("status 200", r.status_code == 200, f"got {r.status_code}")
        count = data.get("count", 0)
        result("count > 0", count > 0, f"count={count}")
        result("video_ids is list", isinstance(data.get("video_ids"), list))
        result("count matches len", count == len(data.get("video_ids", [])))
    except Exception as e:
        result("request failed", False, str(e))


def test_videos_concurrent():
    """GET /videos x10 — no empty results under concurrency."""
    import concurrent.futures

    print("\n[6/10] GET /videos (x10 concurrent)")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(get, "/videos") for _ in range(10)]
            results_list = [f.result() for f in concurrent.futures.as_completed(futures)]

        counts = [r.json().get("count", -1) for r in results_list]
        all_ok = all(c > 0 for c in counts)
        result("all 10 returned data", all_ok, f"counts={counts}")
        result("no empty results", 0 not in counts)
    except Exception as e:
        result("request failed", False, str(e))


def test_frames_lookup():
    """GET /frames/lookup — fetch a keyframe by video_id + timestamp."""
    print("\n[7/10] GET /frames/lookup")
    try:
        r = get("/frames/lookup", params={"video_id": SAMPLE_VIDEO_ID, "timestamp": 10})
        data = r.json()
        if r.status_code == 200:
            result("status 200", True)
            result("has video_id", data.get("video_id") == SAMPLE_VIDEO_ID)
            result("has frames list", isinstance(data.get("frames"), list))
            if data.get("frames"):
                frame = data["frames"][0]
                result("frame has image_base64", "image_base64" in frame)
                result("frame has timestamp_second", "timestamp_second" in frame)
            else:
                result("frames not empty", False, "frames list is empty")
        elif r.status_code in (404, 422):
            result(f"status {r.status_code} (acceptable)", True, "frame/video not found")
        else:
            result(f"unexpected status", False, f"got {r.status_code}: {data}")
    except Exception as e:
        result("request failed", False, str(e))

    # Validation: missing params should return 422
    try:
        r = get("/frames/lookup")
        result("missing params → 422", r.status_code == 422, f"got {r.status_code}")
    except Exception as e:
        result("validation request failed", False, str(e))


def test_transcripts_lookup():
    """GET /transcripts/lookup — fetch transcript for a video."""
    print("\n[8/10] GET /transcripts/lookup")
    try:
        r = get("/transcripts/lookup", params={"video_id": SAMPLE_VIDEO_ID})
        data = r.json()
        if r.status_code == 200:
            result("status 200", True)
            result("has video_id", data.get("video_id") == SAMPLE_VIDEO_ID)
            result("transcript non-empty", bool(data.get("transcript")))
        elif r.status_code == 404:
            result("status 404 (acceptable)", True, "transcript not found")
        else:
            result(f"unexpected status", False, f"got {r.status_code}: {data}")
    except Exception as e:
        result("request failed", False, str(e))

    # Validation: missing params should return 422
    try:
        r = get("/transcripts/lookup")
        result("missing params → 422", r.status_code == 422, f"got {r.status_code}")
    except Exception as e:
        result("validation request failed", False, str(e))


def test_video_catalog():
    """GET /videos/catalog — cached video catalog string."""
    print("\n[9/10] GET /videos/catalog")
    try:
        r = get("/videos/catalog")
        data = r.json()
        if r.status_code == 200:
            result("status 200", True)
            result("has catalog", "catalog" in data and isinstance(data["catalog"], str))
            result("catalog non-empty", bool(data.get("catalog")))
            result("has length_chars", "length_chars" in data)
            result("length_chars matches", data.get("length_chars") == len(data.get("catalog", "")))
        elif r.status_code == 503:
            result("status 503 (acceptable)", True, "catalog not yet generated")
        else:
            result("unexpected status", False, f"got {r.status_code}: {data}")
    except Exception as e:
        result("request failed", False, str(e))


def test_video_catalog_refresh():
    """POST /videos/catalog/refresh — regenerate catalog from Neo4j."""
    print("\n[10/10] POST /videos/catalog/refresh")
    try:
        r = requests.post(f"{BASE_URL}/videos/catalog/refresh", timeout=60)
        data = r.json()
        result("status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            result("has catalog", "catalog" in data and isinstance(data["catalog"], str))
            result("catalog non-empty", bool(data.get("catalog")))
            result("has length_chars", "length_chars" in data)
            result("refreshed is True", data.get("refreshed") is True)
            result("has max_tokens", "max_tokens" in data)
    except Exception as e:
        result("request failed", False, str(e))

    # Test with custom max_tokens
    try:
        r = requests.post(f"{BASE_URL}/videos/catalog/refresh?max_tokens=50", timeout=60)
        data = r.json()
        result("custom max_tokens accepted", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            result("max_tokens echoed", data.get("max_tokens") == 50, f"got {data.get('max_tokens')}")
    except Exception as e:
        result("custom max_tokens request failed", False, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global BASE_URL, SAMPLE_VIDEO_ID

    parser = argparse.ArgumentParser(description="MMCT Docker API integration tests")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the running container (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--video-id",
        default=SAMPLE_VIDEO_ID,
        help=f"Video ID to use for frame/transcript tests (default: {SAMPLE_VIDEO_ID})",
    )
    args = parser.parse_args()
    BASE_URL = args.base_url
    SAMPLE_VIDEO_ID = args.video_id

    print("=" * 60)
    print(f"  MMCT Docker API Tests — {BASE_URL}")
    print("=" * 60)

    # Wait for container to be ready
    print("\nWaiting for container...", end=" ", flush=True)
    for attempt in range(20):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print("ready!")
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        print("TIMEOUT — container not reachable")
        sys.exit(1)

    # Run all tests
    test_root()
    test_health()
    test_providers()
    test_openapi()
    test_videos()
    test_videos_concurrent()
    test_frames_lookup()
    test_transcripts_lookup()
    test_video_catalog()
    test_video_catalog_refresh()

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    if errors:
        print("\n  Failures:")
        for e in errors:
            print(f"    ❌ {e}")
    print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
