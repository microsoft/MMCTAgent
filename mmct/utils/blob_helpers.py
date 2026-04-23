"""Shared utilities for the MCP server and related modules.

Provides common helpers (e.g., video-ID normalization) used across
ingestion upload steps and the MCP data-retrieval routes.
"""

import re

_BLOB_INVALID_RE = re.compile(r'[^a-zA-Z0-9\-_.]')


def normalize_video_id(video_id: str) -> str:
    """Normalize a video ID for use as a blob path segment.

    Replaces any character that is not alphanumeric, ``-``, ``_``, or ``.``
    with an underscore.
    """
    return _BLOB_INVALID_RE.sub('_', video_id)
