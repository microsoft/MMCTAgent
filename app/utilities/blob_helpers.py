"""Shared blob storage helpers.

Provides:
- normalize_video_id(): single source of truth for video ID normalisation
- get_blob_service_client(): singleton async BlobServiceClient (reused across requests)
- blob_exists_cached(): TTL-cached blob existence check
"""

import asyncio
import re
import time
from typing import Optional

from azure.storage.blob.aio import BlobServiceClient
from cachetools import TTLCache
from loguru import logger

from app.config.provider_config import get_settings
from app.config.credentials import resolve_credentials

# ---------------------------------------------------------------------------
# Video ID normalisation
# ---------------------------------------------------------------------------

_BLOB_INVALID_RE = re.compile(r'[^a-zA-Z0-9\-_.]')


def normalize_video_id(video_id: str) -> str:
    """Normalize a video ID for use as a blob path segment.

    Keeps hyphens, underscores, dots, and alphanumerics.
    Replaces any other character with an underscore.
    """
    return _BLOB_INVALID_RE.sub('_', video_id)


# ---------------------------------------------------------------------------
# Singleton BlobServiceClient
# ---------------------------------------------------------------------------

_blob_service_client: Optional[BlobServiceClient] = None
_client_lock = asyncio.Lock()


async def get_blob_service_client() -> BlobServiceClient:
    """Return a shared, long-lived BlobServiceClient.

    The client is created once and reused for all subsequent calls,
    avoiding per-request TCP/TLS/auth overhead.
    """
    global _blob_service_client
    if _blob_service_client is not None:
        return _blob_service_client

    async with _client_lock:
        # Double-check after acquiring lock
        if _blob_service_client is not None:
            return _blob_service_client

        settings = get_settings()
        credentials = resolve_credentials()
        account_url = (
            f"https://{settings.storage_account_name}.blob.core.windows.net"
        )
        _blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credentials
        )
        logger.info("Shared BlobServiceClient initialised")
        return _blob_service_client


def get_account_url() -> str:
    """Return the blob storage account URL (sync, no I/O)."""
    settings = get_settings()
    return f"https://{settings.storage_account_name}.blob.core.windows.net"


# ---------------------------------------------------------------------------
# TTL-cached blob existence check
# ---------------------------------------------------------------------------

# Cache up to 10 000 entries, evict after 5 minutes
_existence_cache: TTLCache = TTLCache(maxsize=10_000, ttl=300)


async def blob_exists_cached(container: str, blob_name: str) -> bool:
    """Check whether a blob exists, with a 5-minute TTL cache.

    Positive *and* negative results are cached so that repeated lookups
    for the same blob don't hit Azure storage.
    """
    cache_key = f"{container}/{blob_name}"
    cached = _existence_cache.get(cache_key)
    if cached is not None:
        return cached

    service = await get_blob_service_client()
    blob_client = service.get_container_client(container).get_blob_client(blob_name)
    try:
        exists = await blob_client.exists()
    except Exception as exc:
        logger.warning(f"Blob existence check failed for {cache_key}: {exc}")
        exists = False
    finally:
        await blob_client.close()

    _existence_cache[cache_key] = exists
    return exists
