"""Video discovery executor — cross-video search.

Only called when video_scope == "cross" (code-enforced).
"""

from typing import Annotated, Dict, List
import asyncio

from loguru import logger

_log = logger.bind(component="state:discovery")


class VideoDiscoveryExecutor:
    """Discovers relevant videos by searching ChapterGroup summaries."""

    def __init__(self, neo4j_provider):
        self.neo4j_provider = neo4j_provider
        self._embedding_provider = None

    def _get_embedding_provider(self):
        """Lazy-load the text embedding provider."""
        if self._embedding_provider is None:
            from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider
            self._embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
        return self._embedding_provider

    async def discover(
        self,
        query: Annotated[str, "Natural language query used for cross-video discovery"],
        limit: Annotated[int, "Maximum videos to return"] = 8,
    ) -> List[tuple[str, float, str]]:
        """Find video IDs relevant to a query using multi-level discovery.

        Searches both ChapterGroup summaries and Chapter-level content,
        then merges results to maximize recall across the video corpus.

        Args:
            query: Natural language query.
            limit: Maximum videos to return.

        Returns:
            List of (video_id, score, video_title) tuples, ranked by score desc.
        """
        _log.info(f"query='{query[:50]}...' limit={limit}")
        try:
            query_embedding = await self._get_embedding_provider().embedding(query)

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

            # Merge: best score per video_id across both levels, collect titles
            video_scores: Dict[str, float] = {}
            video_titles: Dict[str, str] = {}
            for r in cg_results:
                vid = r.get("video_id")
                if vid:
                    video_scores[vid] = max(video_scores.get(vid, 0), r.get("max_score", 0))
                    if r.get("video_title") and vid not in video_titles:
                        video_titles[vid] = r["video_title"]
            for r in ch_results:
                vid = r.get("video_id")
                if vid:
                    video_scores[vid] = max(video_scores.get(vid, 0), r.get("max_score", 0))
                    if r.get("video_title") and vid not in video_titles:
                        video_titles[vid] = r["video_title"]

            # Rank by score, take top-limit — return (video_id, score, title) tuples
            ranked_scores = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            ranked = [
                (vid, score, video_titles.get(vid, ""))
                for vid, score in ranked_scores
            ]

            _log.info(
                f"Found {len(ranked)} videos "
                f"(ChapterGroup: {len(cg_results)}, Chapter: {len(ch_results)}, "
                f"merged: {len(video_scores)})"
            )
            for vid, sc, title in ranked:
                _log.info(f"  {vid} score={sc:.4f} \"{title}\"")
            return ranked

        except Exception as e:
            logger.error(f"Video discovery failed: {e}")
            return []
