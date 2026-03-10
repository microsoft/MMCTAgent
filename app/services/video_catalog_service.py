"""Video Catalog Service — generates a compact video catalog for Planner Agent context.

Fetches ChapterGroup metadata (main_theme, playlist_id, playlist_order) from Neo4j,
groups by playlist, and uses LLM to compress into a token-budgeted catalog string
that is baked into the Planner Agent's system prompt at server startup.
"""

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger


# Module-level singleton: generated once at startup, read by query service
_video_catalog: Optional[str] = None

# Default max token budget for the catalog text (approximate; 1 token ≈ 4 chars)
DEFAULT_CATALOG_MAX_TOKENS = 200


def get_cached_catalog() -> Optional[str]:
    """Return the cached video catalog string, or None if not yet generated."""
    return _video_catalog


async def generate_video_catalog(
    neo4j_provider,
    llm_provider,
    max_tokens: int = DEFAULT_CATALOG_MAX_TOKENS,
) -> str:
    """Generate a compact video catalog and cache it module-wide.

    Steps:
      1. Fetch raw ChapterGroup metadata from Neo4j.
      2. Group by playlist_id (ordered by playlist_order).
      3. LLM-summarize each playlist's themes into a short description.
      4. Orphan videos (no playlist) → compressed "Other Videos" section.
      5. Cap total output to approximately max_tokens.

    Args:
        neo4j_provider: Neo4jQueryProvider instance.
        llm_provider: BaseLLMProvider instance (with chat_completion / generate_json).
        max_tokens: Approximate upper bound on catalog length in tokens.

    Returns:
        The catalog string (also cached in module-level ``_video_catalog``).
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

    # Sort videos inside each playlist by playlist_order, then group_order
    for pid in playlists:
        playlists[pid].sort(key=lambda r: (r.get("playlist_order") or 0, r.get("group_order") or 0))

    # ---- collect unique themes per playlist (preserving order) ----
    playlist_themes: Dict[str, List[str]] = {}
    playlist_video_counts: Dict[str, int] = {}
    for pid, rows in playlists.items():
        seen_themes: set = set()
        themes: list = []
        video_ids: set = set()
        for r in rows:
            video_ids.add(r["video_id"])
            theme = r.get("main_theme", "")
            if theme and theme not in seen_themes:
                seen_themes.add(theme)
                themes.append(theme)
        playlist_themes[pid] = themes
        playlist_video_counts[pid] = len(video_ids)

    # Collect orphan themes
    orphan_themes: List[str] = []
    orphan_video_ids: set = set()
    seen_orphan: set = set()
    for r in orphans:
        orphan_video_ids.add(r["video_id"])
        theme = r.get("main_theme", "")
        if theme and theme not in seen_orphan:
            seen_orphan.add(theme)
            orphan_themes.append(theme)

    # ---- build LLM prompt to compress themes ----
    max_chars = max_tokens * 4  # rough token-to-char estimate

    sections = []
    for pid, themes in playlist_themes.items():
        n = playlist_video_counts[pid]
        themes_str = ", ".join(themes[:30])  # cap input length
        sections.append(
            f"Playlist {pid} ({n} videos): {themes_str}"
        )

    if orphan_themes:
        themes_str = ", ".join(orphan_themes[:15])
        sections.append(
            f"Other videos ({len(orphan_video_ids)} videos, no playlist): {themes_str}"
        )

    raw_input = "\n".join(sections)

    orphan_instruction = (
        "\n- For 'Other videos', add a brief note about what they cover." if orphan_themes else ""
    )
    prompt = (
        "You are creating a video corpus description for an AI query planner. "
        "The planner uses this to understand what content is available before formulating "
        "search queries against a video knowledge graph.\n\n"
        "Below are playlists of educational videos with their topic themes (in curriculum order).\n\n"
        f"{raw_input}\n\n"
        f"Write a descriptive catalog of AT MOST {max_chars} characters that:\n"
        "- Gives each playlist a clear, descriptive name (not the raw ID)\n"
        "- Describes the subject area, scope, and progression of each playlist\n"
        "- Highlights the key topics and concepts covered\n"
        "- Uses multiple lines per playlist if needed for clarity\n"
        f"- Includes the video count per playlist{orphan_instruction}\n\n"
        "The goal is to help the planner quickly determine which playlist(s) are relevant "
        "to a user's query. Use plain text, no markdown. "
        "Do NOT add sections that are not in the input.\n\n"
        "Respond with ONLY the catalog text, nothing else."
    )

    # logger.info(f"Video catalog LLM prompt:\n{prompt}")

    try:
        response = await llm_provider.chat_completion(
            messages=[{"role": "user", "content": prompt}],
        )
        catalog_text = response.get("content", "").strip()
    except Exception as e:
        logger.error(f"LLM catalog generation failed, falling back to raw themes: {e}")
        # Fallback: just list playlist IDs with video counts
        lines = []
        for pid, themes in playlist_themes.items():
            n = playlist_video_counts[pid]
            top = ", ".join(themes[:5])
            lines.append(f"Playlist ({n} videos): {top}")
        if orphan_themes:
            lines.append(f"Other ({len(orphan_video_ids)} videos): {', '.join(orphan_themes[:3])}")
        catalog_text = "\n".join(lines)
        if len(catalog_text) > max_chars:
            catalog_text = catalog_text[:max_chars].rsplit(" ", 1)[0] + "..."

    _video_catalog = catalog_text
    logger.info(f"Video catalog generated ({len(catalog_text)} chars, ~{len(catalog_text)//4} tokens)")
    return _video_catalog
