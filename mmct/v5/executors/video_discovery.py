"""Video discovery executor — cross-video search.

Replaces V4's find_relevant_videos tool wrapper.
Only called when video_scope == "cross" (code-enforced).
"""

from typing import Any, Dict, List
from datetime import datetime

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
        limit: int = 3,
    ) -> List[str]:
        """Find video IDs relevant to a query.

        Args:
            query: Natural language query.
            limit: Maximum videos to return.

        Returns:
            List of relevant video_id strings.
        """
        _log(f"query='{query[:50]}...' limit={limit}")
        try:
            query_embedding = await self.embedding_provider.embedding(query)
            results = await self.neo4j_provider.find_relevant_videos(
                query_embedding=query_embedding,
                limit=limit,
            )

            video_ids = [r.get("video_id") for r in results if r.get("video_id")]
            _log(f"{_YELLOW}Found {len(video_ids)} relevant videos: {video_ids}{_RESET}")
            return video_ids

        except Exception as e:
            logger.error(f"Video discovery failed: {e}")
            return []
