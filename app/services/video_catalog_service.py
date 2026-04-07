"""Video catalog service — builds a compact corpus description for the query agent.

At server startup, this service fetches ChapterGroup metadata from the knowledge
graph, groups videos by playlist, and uses an LLM to compress the raw theme data
into a token-budgeted catalog string.  The catalog is baked into the query
agent's system prompt so the planner knows what content is available before
formulating search queries.

The generated catalog is cached module-wide and can be refreshed on demand via
the /videos/catalog/refresh endpoint.
"""

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger


# Module-level cache: generated once at startup, updated on refresh
_video_catalog: Optional[str] = None

# Approximate token budget for the catalog (1 token ≈ 4 characters)
DEFAULT_CATALOG_MAX_TOKENS = 200


def get_cached_catalog() -> Optional[str]:
    """Return the cached catalog string, or None if not yet generated."""
    return _video_catalog


async def generate_video_catalog(
    neo4j_provider,
    llm_provider,
    max_tokens: int = DEFAULT_CATALOG_MAX_TOKENS,
) -> str:
    """Generate a compact video catalog and store it in the module-level cache.

    Fetches ChapterGroup metadata from the knowledge graph, groups entries by
    playlist, and calls the LLM to produce a human-readable, token-budgeted
    description of the available video corpus.

    Steps:
      1. Fetch raw ChapterGroup rows from the graph (video_id, main_theme,
         playlist_id, playlist_order, group_order).
      2. Separate entries into playlist buckets and orphan (no playlist) list.
      3. Collect unique themes per playlist in curriculum order.
      4. Ask the LLM to compress the theme lists into a descriptive catalog.
      5. Fall back to a plain-text summary if the LLM call fails.
      6. Cache the result and return it.

    Args:
        neo4j_provider: Knowledge-graph query provider with a
            ``get_video_catalog_raw()`` method.
        llm_provider: LLM provider with a ``chat_completion()`` method.
        max_tokens: Approximate upper bound on catalog length in tokens.

    Returns:
        The catalog string (also stored in the module-level cache).
    """
    global _video_catalog

    raw = await neo4j_provider.get_video_catalog_raw()
    if not raw:
        logger.warning("No ChapterGroup metadata found — video catalog will be empty.")
        _video_catalog = ""
        return _video_catalog

    # ---- group by playlist ----
    playlists: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    orphans: List[Dict[str, Any]] = []

    for row in raw:
        pid = row.get("playlist_id")
        if pid:
            playlists[pid].append(row)
        else:
            orphans.append(row)

    for pid in playlists:
        playlists[pid].sort(
            key=lambda r: (r.get("playlist_order") or 0, r.get("group_order") or 0)
        )

    # ---- collect unique themes per playlist ----
    playlist_themes: Dict[str, List[str]] = {}
    playlist_video_counts: Dict[str, int] = {}
    for pid, rows in playlists.items():
        seen: set = set()
        themes: list = []
        video_ids: set = set()
        for r in rows:
            video_ids.add(r["video_id"])
            theme = r.get("main_theme", "")
            if theme and theme not in seen:
                seen.add(theme)
                themes.append(theme)
        playlist_themes[pid] = themes
        playlist_video_counts[pid] = len(video_ids)

    orphan_themes: List[str] = []
    orphan_video_ids: set = set()
    seen_orphan: set = set()
    for r in orphans:
        orphan_video_ids.add(r["video_id"])
        theme = r.get("main_theme", "")
        if theme and theme not in seen_orphan:
            seen_orphan.add(theme)
            orphan_themes.append(theme)

    # ---- build LLM prompt ----
    max_chars = max_tokens * 4
    sections = []
    for pid, themes in playlist_themes.items():
        n = playlist_video_counts[pid]
        sections.append(f"Playlist {pid} ({n} videos): {', '.join(themes[:30])}")
    if orphan_themes:
        sections.append(
            f"Other videos ({len(orphan_video_ids)} videos, no playlist): "
            f"{', '.join(orphan_themes[:15])}"
        )

    raw_input = "\n".join(sections)
    orphan_note = (
        "\n- For 'Other videos', add a brief note about what they cover."
        if orphan_themes
        else ""
    )

    prompt = (
        "You are creating a video corpus description for an AI query planner. "
        "The planner uses this to understand what content is available before "
        "formulating search queries against a video knowledge graph.\n\n"
        "Below are playlists of educational videos with their topic themes "
        "(in curriculum order).\n\n"
        f"{raw_input}\n\n"
        f"Write a descriptive catalog of AT MOST {max_chars} characters that:\n"
        "- Gives each playlist a clear, descriptive name (not the raw ID)\n"
        "- Describes the subject area, scope, and progression of each playlist\n"
        "- Highlights the key topics and concepts covered\n"
        "- Uses multiple lines per playlist if needed for clarity\n"
        f"- Includes the video count per playlist{orphan_note}\n\n"
        "The goal is to help the planner quickly determine which playlist(s) are "
        "relevant to a user's query. Use plain text, no markdown. "
        "Do NOT add sections that are not in the input.\n\n"
        "Respond with ONLY the catalog text, nothing else."
    )

    try:
        response = await llm_provider.chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )
        catalog_text = response.get("content", "").strip()
    except Exception as exc:
        logger.error(f"LLM catalog generation failed, falling back to raw themes: {exc}")
        lines = []
        for pid, themes in playlist_themes.items():
            n = playlist_video_counts[pid]
            lines.append(f"Playlist ({n} videos): {', '.join(themes[:5])}")
        if orphan_themes:
            lines.append(
                f"Other ({len(orphan_video_ids)} videos): {', '.join(orphan_themes[:3])}"
            )
        catalog_text = "\n".join(lines)
        if len(catalog_text) > max_chars:
            catalog_text = catalog_text[:max_chars].rsplit(" ", 1)[0] + "..."

    _video_catalog = catalog_text
    logger.info(
        f"Video catalog generated ({len(catalog_text)} chars, "
        f"~{len(catalog_text) // 4} tokens)"
    )
    return _video_catalog
