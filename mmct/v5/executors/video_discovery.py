"""Video discovery executor — cross-video search.

Replaces V4's find_relevant_videos tool wrapper.
Only called when video_scope == "cross" (code-enforced).
"""

from typing import Any, Dict, List
from datetime import datetime
import asyncio

from loguru import logger


_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GRAY = "\033[90m"
_RESET = "\033[0m"


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{_GRAY}[{now}]{_RESET} {_CYAN}[V5:discovery]{_RESET} {msg}", flush=True)


class VideoDiscoveryExecutor:
    """Discovers relevant videos by searching ChapterGroup summaries."""

    def __init__(self, neo4j_provider, embedding_provider):
        self.neo4j_provider = neo4j_provider
        self.embedding_provider = embedding_provider

    async def discover(
        self,
        query: str,
        limit: int = 8,
    ) -> List[str]:
        """Find video IDs relevant to a query using multi-level discovery.

        Searches both ChapterGroup summaries and Chapter-level content,
        then merges results to maximize recall across the video corpus.

        Args:
            query: Natural language query.
            limit: Maximum videos to return.

        Returns:
            List of relevant video_id strings, ranked by best score.
        """
        _log(f"query='{query[:50]}...' limit={limit}")
        try:
            query_embedding = await self.embedding_provider.embedding(query)

            # Multi-level discovery: search ChapterGroups AND Chapters in parallel
            cg_task = self.neo4j_provider.find_relevant_videos(
                query_embedding=query_embedding,
                limit=limit,
            )
            ch_task = self.neo4j_provider.find_relevant_videos_by_chapter(
                query_embedding=query_embedding,
                limit=limit,
            )
            cg_results, ch_results = await asyncio.gather(cg_task, ch_task)

            # Merge: best score per video_id across both levels
            video_scores: Dict[str, float] = {}
            for r in cg_results:
                vid = r.get("video_id")
                if vid:
                    video_scores[vid] = max(video_scores.get(vid, 0), r.get("max_score", 0))
            for r in ch_results:
                vid = r.get("video_id")
                if vid:
                    video_scores[vid] = max(video_scores.get(vid, 0), r.get("max_score", 0))

            # Rank by score, take top-limit
            ranked = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            video_ids = [vid for vid, _ in ranked]

            _log(
                f"{_YELLOW}Found {len(video_ids)} videos "
                f"(ChapterGroup: {len(cg_results)}, Chapter: {len(ch_results)}, "
                f"merged: {len(video_scores)}){_RESET}"
            )
            return video_ids

        except Exception as e:
            logger.error(f"Video discovery failed: {e}")
            return []
