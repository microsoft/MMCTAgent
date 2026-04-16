"""Retrieval executor — programmatic graph search and traversal.

Replaces the graph pipeline's VideoAgent LLM. All tool calls are code-driven:
parallel search_graph per sub-query, overview, traverse.
"""

import asyncio
import json
from typing import Annotated, Any, Dict, List, Optional, Tuple

from loguru import logger

from mmct.video_pipeline.core.graph import node_registry
from mmct.video_pipeline.utils import OutputFormatterMixin

_log = logger.bind(component="state:retrieval")


class RetrievalExecutor(OutputFormatterMixin):
    """Executes graph retrieval programmatically. No LLM needed.

    Wraps Neo4jQueryProvider and embedding_provider to run
    search_graph, get_video_overview, traverse_graph, and
    search_keyframes — all driven by the validated plan.
    """

    def __init__(self, neo4j_provider):
        self.neo4j_provider = neo4j_provider
        self._embedding_provider = None
        self._image_embedding_provider = None

    def _get_embedding_provider(self):
        """Lazy-load the text embedding provider."""
        if self._embedding_provider is None:
            from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider
            self._embedding_provider = FastEmbedBGEsmallEmbeddingProvider()
        return self._embedding_provider

    def _get_image_embedding_provider(self):
        """Lazy-load the image embedding provider."""
        if self._image_embedding_provider is None:
            from mmct.providers.custom_providers import FastEmbedQdrantCLIPEmbeddingProvider
            self._image_embedding_provider = FastEmbedQdrantCLIPEmbeddingProvider()
        return self._image_embedding_provider

    async def search(
        self,
        sub_queries: Annotated[List[str], "Focused sub-queries to embed and search"],
        targets: Annotated[List[str], "Graph node types to search"],
        video_ids: Annotated[Optional[List[str]], "Optional video scope filter"] = None,
        limit: Annotated[int, "Maximum results per target and sub-query"] = 5,
    ) -> List[Dict[str, Any]]:
        """Run parallel vector searches — one per sub-query.

        Args:
            sub_queries: List of focused query strings.
            targets: Node types to search.
            video_ids: Optional video ID filter.
            limit: Max results per target per sub-query.

        Returns:
            Merged list of result dicts (deduplicated by node_id).
        """
        _log.info(f"targets={targets} sub_queries={len(sub_queries)} video_ids={video_ids}")

        # Validate targets
        valid = set(node_registry.names())
        invalid = set(targets) - valid
        if invalid:
            logger.error(f"Invalid targets: {invalid}. Valid: {valid}")
            return []

        # Parallel: embed all sub-queries at once
        embeddings = await asyncio.gather(
            *[self._get_embedding_provider().embedding(sq) for sq in sub_queries]
        )

        # Parallel: run search for each sub-query (hybrid: vector + keyword)
        search_tasks = [
            self.neo4j_provider.search_multiple_granularities(
                query_embedding=emb,
                targets=targets,
                video_ids=video_ids,
                limit_per_type=limit,
                query_text=sq,
            )
            for emb, sq in zip(embeddings, sub_queries)
        ]
        all_results = await asyncio.gather(*search_tasks)

        # Merge and deduplicate
        return self._merge_results(all_results, targets)

    async def overview(
        self,
        video_id: Annotated[str, "Video identifier to summarize"],
        level: Annotated[str, "Node type to fetch for overview"] = "ChapterGroup",
        limit: Annotated[int, "Maximum nodes to return"] = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch all nodes for a video (no vector search)."""
        _log.info(f"video_id={video_id} level={level}")

        results = await self.neo4j_provider.get_all_nodes_for_video(
            video_id=video_id,
            node_type=level,
            limit=limit,
        )
        return self._format_node_results(results, level)

    async def traverse(
        self,
        node_ids: Annotated[List[str], "Source node identifiers"],
        target: Annotated[str, "Target node type to traverse to"],
        video_id: Annotated[Optional[str], "Optional single-video filter"] = None,
        limit: Annotated[int, "Maximum traversed nodes to return"] = 20,
    ) -> List[Dict[str, Any]]:
        """Traverse graph relationships."""
        _log.info(f"nodes={node_ids[:3]} → {target}")

        results = await self.neo4j_provider.traverse_relationships(
            source_ids=node_ids,
            target_type=target,
            video_id=video_id,
            limit=limit,
        )
        return self._format_node_results(results, target)

    async def get_sibling_chapters(
        self,
        chapter_ids: Annotated[List[str], "Chapter node identifiers"],
        limit: Annotated[int, "Maximum sibling chapters to return"] = 20,
    ) -> List[Dict[str, Any]]:
        """Get all sibling chapters sharing the same parent ChapterGroup."""
        _log.info(f"chapters={chapter_ids[:3]} → sibling Chapters")

        results = await self.neo4j_provider.get_sibling_chapters(
            chapter_ids=chapter_ids,
            limit=limit,
        )
        return self._format_node_results(results, "Chapter")

    async def search_keyframes(
        self,
        query: Annotated[str, "Visual search query"],
        video_ids: Annotated[Optional[List[str]], "Optional multi-video scope filter"] = None,
        limit: Annotated[int, "Maximum keyframes to return"] = 5,
    ) -> List[Dict[str, Any]]:
        """Search keyframes by image embedding vector search."""
        _log.info(f"query='{query[:50]}' video_ids={video_ids}")

        query_embedding = await self._get_image_embedding_provider().text_embedding(query)
        results = await self.neo4j_provider.search_keyframes(
            query_embedding=query_embedding,
            video_ids=video_ids,
            limit=limit,
        )

        keyframes = []
        for kf in results:
            if hasattr(kf, "to_dict"):
                kf_dict = kf.to_dict()
            else:
                kf_dict = kf
            props = kf_dict.get("properties", kf_dict)
            keyframes.append({
                "keyframe_id": kf_dict.get("node_id") or props.get("node_id"),
                "blob_url": props.get("blob_url"),
                "timestamp": props.get("timestamp"),
                "video_id": props.get("video_id"),
                "chapter_id": props.get("chapter_id"),
                "score": round(kf_dict.get("score", 0), 4),
            })

        _log.info(f"Found {len(keyframes)} keyframes")
        return keyframes

    def _merge_results(
        self,
        all_results: List[Dict[str, list]],
        targets: List[str],
    ) -> List[Dict[str, Any]]:
        """Merge, deduplicate, and diversify results from multiple sub-query searches.

        Guarantees video diversity: reserves the top-scoring result from each
        discovered video, then fills the remaining slots by global score.
        """
        seen_ids = set()
        merged = []

        for result_by_type in all_results:
            for granularity, items in result_by_type.items():
                if not items:
                    continue

                node_type = node_registry.get(granularity)

                for item in items:
                    if hasattr(item, "to_dict"):
                        item_dict = item.to_dict()
                    else:
                        item_dict = item

                    node_id = item_dict.get("node_id") or item_dict.get("id")
                    if node_id in seen_ids:
                        continue
                    seen_ids.add(node_id)

                    props = item_dict.get("properties", item_dict)
                    entry = {
                        "node_id": node_id,
                        "node_type": granularity,
                        "video_id": item_dict.get("video_id") or props.get("video_id"),
                        "score": round(item_dict.get("score", 0), 4),
                    }
                    if node_type:
                        entry.update(node_type.format_search_result(props))

                    merged.append(entry)

        # Video diversity: ensure every discovered video gets representation
        merged = self._diversify_by_video(merged)

        total = len(merged)
        _log.info(f"Found {total} results (deduplicated, diversified)")
        for i, r in enumerate(merged):
            vid = r.get("video_id", "?")
            ntype = r.get("node_type", "?")
            score = r.get("score", 0)
            st = r.get("start_time", "")
            et = r.get("end_time", "")
            title = r.get("video_title", "")
            desc = r.get("summary") or r.get("description") or r.get("transcript") or ""
            time_range = f" [{st}-{et}s]" if st != "" else ""
            title_str = f" ({title})" if title else ""
            _log.debug(f"  [{i+1}] {ntype} {vid}{time_range} score={score:.4f}{title_str}")
            if desc:
                _log.debug(f"       {desc}")
        return merged

    @staticmethod
    def _diversify_by_video(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank results to guarantee min-1-per-video representation.

        Strategy:
        1. Group results by video_id
        2. Pick the top-scoring result from each video (reserved slots)
        3. Fill remaining slots from the rest, sorted by score
        4. Return reserved + remainder
        """
        if not results:
            return results

        from collections import defaultdict
        by_video: dict = defaultdict(list)
        for r in results:
            vid = r.get("video_id") or "unknown"
            by_video[vid].append(r)

        # Sort each video's results by score descending
        for vid in by_video:
            by_video[vid].sort(key=lambda x: x.get("score", 0), reverse=True)

        # Reserve top-1 per video
        reserved = []
        remainder = []
        for vid, items in by_video.items():
            reserved.append(items[0])
            remainder.extend(items[1:])

        # Sort reserved by score, remainder by score
        reserved.sort(key=lambda x: x.get("score", 0), reverse=True)
        remainder.sort(key=lambda x: x.get("score", 0), reverse=True)

        return reserved + remainder

    def _format_node_results(self, results: list, node_type_name: str) -> List[Dict[str, Any]]:
        """Format raw node results with registry-based formatting."""
        node_type = node_registry.get(node_type_name)
        formatted = []

        for item in results:
            if hasattr(item, "properties"):
                props = item.properties
                node_id = item.node_id
            elif hasattr(item, "to_dict"):
                d = item.to_dict()
                props = d.get("properties", d)
                node_id = d.get("node_id")
            else:
                props = item
                node_id = item.get("node_id")

            entry = {"node_id": node_id, "video_id": props.get("video_id")}
            if node_type:
                entry.update(node_type.format_search_result(props))
            formatted.append(entry)

        return formatted
