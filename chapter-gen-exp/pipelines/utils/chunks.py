"""Utilities for handling chunk metadata in pipeline steps."""
from __future__ import annotations

from typing import Dict, List, Optional, Any


def resolve_chunks(
    data_store: Any,
    chunks_step: Optional[str],
    fallback_duration: float,
) -> List[Dict[str, float]]:
    """Retrieve chunk definitions from a prior step, or fallback to full video."""

    chunks: List[Dict[str, float]] = []
    if chunks_step:
        bucket = data_store.get(chunks_step)
        raw_chunks = None
        if bucket is not None:
            getter = getattr(bucket, "get", None)
            if callable(getter):
                raw_chunks = getter("chunks")
            elif isinstance(bucket, dict):  # pragma: no cover - defensive
                raw_chunks = bucket.get("chunks")
        if raw_chunks:
            for idx, chunk in enumerate(raw_chunks):
                start = float(chunk.get("start", 0.0))
                end = float(chunk.get("end", fallback_duration))
                if end <= start:
                    continue
                chunks.append(
                    {
                        "index": int(chunk.get("index", idx)),
                        "start": start,
                        "end": end,
                    }
                )
    if not chunks:
        chunks = [
            {
                "index": 0,
                "start": 0.0,
                "end": float(fallback_duration),
            }
        ]
    chunks.sort(key=lambda item: item["start"])
    return chunks
