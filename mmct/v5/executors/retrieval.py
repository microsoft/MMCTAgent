"""Retrieval executor — programmatic graph search and traversal.

Replaces V4's VideoAgent LLM. All tool calls are code-driven:
parallel search_graph per sub-query, overview, traverse.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from loguru import logger

from mmct.graph import node_registry
from mmct.v5.utils import OutputFormatterMixin


_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_GRAY = "\033[90m"
_RESET = "\033[0m"


def _log(tool: str, msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{_GRAY}[{now}]{_RESET} {_CYAN}[V5:{tool}]{_RESET} {msg}", flush=True)


class RetrievalExecutor(OutputFormatterMixin):
    """Executes graph retrieval programmatically. No LLM needed.

    Wraps Neo4jQueryProvider and embedding_provider to run
    search_graph, get_video_overview, traverse_graph, and
    search_keyframes — all driven by the validated plan.
    """

    def __init__(self, neo4j_provider, embedding_provider, image_embedding_provider=None):
        self.neo4j_provider = neo4j_provider
        self.embedding_provider = embedding_provider
        self.image_embedding_provider = image_embedding_provider

    async def search(
        self,
        sub_queries: List[str],
        targets: List[str],
        video_ids: Optional[List[str]] = None,
        limit: int = 5,
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
        _log("search", f"targets={targets} sub_queries={len(sub_queries)} video_ids={video_ids}")

        # Validate targets
        valid = set(node_registry.names())
        invalid = set(targets) - valid
        if invalid:
            logger.error(f"Invalid targets: {invalid}. Valid: {valid}")
            return []

        # Parallel: embed all sub-queries at once
        embeddings = await asyncio.gather(
            *[self.embedding_provider.embedding(sq) for sq in sub_queries]
        )

        # Parallel: run search for each sub-query
        search_tasks = [
            self.neo4j_provider.search_multiple_granularities(
                query_embedding=emb,
                targets=targets,
                video_ids=video_ids,
                limit_per_type=limit,
            )
            for emb in embeddings
        ]
        all_results = await asyncio.gather(*search_tasks)

        # Merge and deduplicate
        return self._merge_results(all_results, targets)

    async def overview(
        self,
        video_id: str,
        level: str = "ChapterGroup",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch all nodes for a video (no vector search)."""
        _log("overview", f"video_id={video_id} level={level}")

        results = await self.neo4j_provider.get_all_nodes_for_video(
            video_id=video_id,
            node_type=level,
            limit=limit,
        )
        return self._format_node_results(results, level)

    async def traverse(
        self,
        node_ids: List[str],
        target: str,
        video_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Traverse graph relationships."""
        _log("traverse", f"nodes={node_ids[:3]} → {target}")

        results = await self.neo4j_provider.traverse_relationships(
            source_ids=node_ids,
            target_type=target,
            video_id=video_id,
            limit=limit,
        )
        return self._format_node_results(results, target)

    async def search_keyframes(
        self,
        query: str,
        video_ids: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search keyframes by image embedding vector search."""
        _log("search_keyframes", f"query='{query[:50]}' video_ids={video_ids}")

        if self.image_embedding_provider is None:
            logger.warning("Image embedding provider not configured")
            return []

        query_embedding = await self.image_embedding_provider.text_embedding(query)
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

        _log("search_keyframes", f"{_YELLOW}Found {len(keyframes)} keyframes{_RESET}")
        return keyframes

    def _merge_results(
        self,
        all_results: List[Dict[str, list]],
        targets: List[str],
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from multiple sub-query searches."""
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

        total = len(merged)
        _log("search", f"{_YELLOW}Found {total} results (deduplicated){_RESET}")
        return merged

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
