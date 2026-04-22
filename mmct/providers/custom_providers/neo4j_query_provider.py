"""Unified Neo4j Query Provider for v4 and v5 pipelines.

Concrete implementation of BaseNeo4jQueryProvider covering the full feature
set of both pipelines:
- v4: vector search, hybrid keyword search, graph traversal, cross-video discovery
- v5: all of the above + chapter-level discovery, sibling expansion, video titles

Key design principles:
- NEVER return embedding vectors in query results (latency + context bloat)
- Explicit property selection in all Cypher queries
- Post-filtering by video_id and time_range (applied AFTER HNSW search, with 3x over-fetch)
- Async execution with connection pooling
- HNSW algorithm with configurable ef_search for recall/latency tradeoff

NOTE: Neo4j's db.index.vector.queryNodes does NOT support pre-filtering.
Filters are applied post-search, so we over-fetch and then filter.
"""

import asyncio
import json as _json
from typing import List, Dict, Any, Optional, Tuple

from loguru import logger

from mmct.video_pipeline.core.graph import node_registry
from mmct.providers.base.graph_query_provider import BaseGraphQueryProvider, SearchResult


# Default HNSW search parameter
# Higher ef_search = better recall but higher latency
# Recommended: ef_search >= k (limit) for good results
DEFAULT_EF_SEARCH = 100

# Fulltext index names for keyword search (Lucene-based)
FULLTEXT_INDEX_MAP = {
    "Chapter": "chapter_fulltext_index",
    "ChapterGroup": "chaptergroup_fulltext_index",
}


class Neo4jQueryProvider(BaseGraphQueryProvider):
    """Unified Neo4j query provider for v4 and v5 pipelines using HNSW vector search.

    Provides optimized HNSW vector search, hybrid keyword search, and graph
    traversal methods for multi-granularity retrieval from the video knowledge graph.

    All queries explicitly select properties to avoid returning embeddings.
    Uses HNSW (Hierarchical Navigable Small World) algorithm for ANN search.

    Attributes:
        uri: Neo4j connection URI.
        database: Database name.
        ef_search: HNSW ef parameter for search (higher = better recall, higher latency).
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        ef_search: int = DEFAULT_EF_SEARCH,
        max_connection_pool_size: int = 100,
        connection_acquisition_timeout: int = 60,
        max_connection_lifetime: int = 3600,
        keep_alive: bool = True,
    ):
        """Initialize Neo4j query provider.

        Args:
            uri: Neo4j connection URI (bolt://host:port).
            username: Neo4j username.
            password: Neo4j password.
            database: Database name (default "neo4j").
            ef_search: HNSW search parameter (default 100). Higher values give
                better recall but increase latency. Should be >= limit for good results.
            max_connection_pool_size: Maximum pooled connections for concurrent reads.
            connection_acquisition_timeout: Seconds to wait for a pooled connection.
            max_connection_lifetime: Maximum lifetime of a pooled connection in seconds.
            keep_alive: Enable TCP keep-alive to detect stale connections.
        """
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._ef_search = ef_search
        self._max_connection_pool_size = max_connection_pool_size
        self._connection_acquisition_timeout = connection_acquisition_timeout
        self._max_connection_lifetime = max_connection_lifetime
        self._keep_alive = keep_alive
        self._driver = None
        self._init_lock = asyncio.Lock()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def check_health(self) -> Dict[str, Any]:
        """Verify Neo4j connectivity with an explicit driver ping.

        Uses ``driver.verify_connectivity()`` which performs a real TCP
        round-trip, unlike ``get_all_video_ids()`` which may silently
        return empty results for unreachable hosts.
        """
        try:
            await self._ensure_driver()
            await self._driver.verify_connectivity()
        except Exception as e:
            return {"status": "error", "error": str(e)}

        try:
            ids = await self.get_all_video_ids()
            return {"status": "ok", "video_count": len(ids)}
        except Exception as e:
            return {"status": "error", "error": f"connected but query failed: {e}"}

    async def _ensure_driver(self) -> None:
        """Lazy initialize Neo4j async driver (double-checked locking)."""
        if self._driver is not None:
            return
        async with self._init_lock:
            if self._driver is not None:
                return
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self._uri,
                    auth=(self._username, self._password),
                    max_connection_pool_size=self._max_connection_pool_size,
                    connection_acquisition_timeout=self._connection_acquisition_timeout,
                    max_connection_lifetime=self._max_connection_lifetime,
                    keep_alive=self._keep_alive,
                )
                logger.info(
                    f"Neo4jQueryProvider async driver connected to {self._uri} "
                    f"(pool_size={self._max_connection_pool_size})"
                )
            except ImportError:
                raise ImportError(
                    "neo4j package required. Install with: pip install neo4j"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    async def _run_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a read query using the async driver and connection pool.

        Each call acquires a connection from the pool, executes the query,
        and returns the connection — enabling high concurrency for reads.

        Retries once on closed transport errors (stale pooled connections).

        Args:
            query: Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of record dictionaries.
        """
        await self._ensure_driver()
        db = self.get_database(self._database)
        for attempt in range(2):
            try:
                async with self._driver.session(database=db) as session:
                    result = await session.run(query, parameters or {})
                    return [dict(record) async for record in result]
            except Exception as e:
                is_closed_transport = "handler is closed" in str(e) or "closed=True" in str(e)
                if attempt == 0 and is_closed_transport:
                    logger.warning(f"Stale Neo4j connection detected, retrying: {e}")
                    continue
                raise

    async def close(self) -> None:
        """Close the Neo4j async driver and release pooled connections."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4jQueryProvider connection closed")

    # =========================================================================
    # Granular Vector Search
    # =========================================================================

    async def search_chapter_groups(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        return await self._vector_search(
            node_type="ChapterGroup",
            query_embedding=query_embedding,
            video_ids=video_ids,
            time_range=time_range,
            limit=limit,
        )

    async def search_chapters(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        return await self._vector_search(
            node_type="Chapter",
            query_embedding=query_embedding,
            video_ids=video_ids,
            time_range=time_range,
            limit=limit,
        )

    async def search_events(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        return await self._vector_search(
            node_type="Event",
            query_embedding=query_embedding,
            video_ids=video_ids,
            time_range=time_range,
            limit=limit,
        )

    async def search_objects(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        return await self._vector_search(
            node_type="Object",
            query_embedding=query_embedding,
            video_ids=video_ids,
            time_range=time_range,
            limit=limit,
        )

    async def search_keyframes(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search Keyframe nodes by image embedding similarity.

        Note: Keyframes use 512-dim image embeddings (QdrantCLIP), not the
        384-dim text embeddings used by other node types.
        """
        return await self._vector_search(
            node_type="Keyframe",
            query_embedding=query_embedding,
            video_ids=video_ids,
            time_range=time_range,
            limit=limit,
        )

    async def search_multiple_granularities(
        self,
        query_embedding: List[float],
        targets: List[str],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit_per_type: int = 5,
        sort_by_time: bool = False,
        query_text: Optional[str] = None,
    ) -> Dict[str, List[SearchResult]]:
        """Search multiple node types in parallel with hybrid vector + keyword search."""
        valid_targets = [
            t for t in targets
            if node_registry.get(t) and node_registry.get(t).is_searchable
        ]

        if not valid_targets:
            return {}

        tasks = []
        task_labels = []

        for target in valid_targets:
            tasks.append(
                self._vector_search(
                    node_type=target,
                    query_embedding=query_embedding,
                    video_ids=video_ids,
                    time_range=time_range,
                    limit=limit_per_type,
                    sort_by_time=sort_by_time,
                )
            )
            task_labels.append(("vector", target))

        # Add keyword search for types that have fulltext indexes
        if query_text:
            for target in valid_targets:
                if target in FULLTEXT_INDEX_MAP:
                    tasks.append(
                        self._keyword_search(
                            node_type=target,
                            query_text=query_text,
                            video_ids=video_ids,
                            limit=limit_per_type,
                        )
                    )
                    task_labels.append(("keyword", target))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        vector_by_type: Dict[str, List[SearchResult]] = {}
        keyword_by_type: Dict[str, List[SearchResult]] = {}

        for (search_type, target), result in zip(task_labels, results):
            if isinstance(result, Exception):
                logger.error(f"Error in {search_type} search for {target}: {result}")
                continue
            if search_type == "vector":
                vector_by_type[target] = result
            else:
                keyword_by_type[target] = result

        output = {}
        for target in valid_targets:
            vec = vector_by_type.get(target, [])
            kw = keyword_by_type.get(target, [])
            if kw:
                merged = self._merge_search_results(vec, kw)
                output[target] = merged[:limit_per_type * 2]
            else:
                output[target] = vec

        return output

    # =========================================================================
    # Internal Search Helpers
    # =========================================================================

    async def _vector_search(
        self,
        node_type: str,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
        sort_by_time: bool = False,
    ) -> List[SearchResult]:
        """Internal HNSW vector search implementation with post-filtering.

        Uses Neo4j's HNSW index with optional post-filtering by video_id and
        time_range. Since Neo4j vector search doesn't support pre-filtering,
        we over-fetch (3x limit) and apply filters after the vector search.

        Keyframes (30k+) need a much larger over-fetch than text nodes (~200-2k)
        when filtering by video_id, because each video is a small fraction of total.
        """
        await self._ensure_driver()

        nt = node_registry.get(node_type)
        if not nt:
            logger.warning(f"Unknown node type: {node_type}")
            return []

        index_name = nt.embedding_index_name
        properties = nt.neo4j_properties
        prop_return = ", ".join(f"node.{p} AS {p}" for p in properties)

        params = {
            "index_name": index_name,
            "embedding": query_embedding,
            "limit": limit,
        }

        post_filter_conditions = []

        if video_ids:
            post_filter_conditions.append("node.video_id IN $video_ids")
            params["video_ids"] = video_ids

        if time_range:
            time_start, time_end = time_range
            params["time_start"] = time_start
            params["time_end"] = time_end

            if node_type in ("ChapterGroup", "Chapter", "Transcript"):
                post_filter_conditions.append("node.start_time < $time_end")
                post_filter_conditions.append("node.end_time > $time_start")
            elif node_type in ("Event", "Keyframe"):
                post_filter_conditions.append("node.timestamp >= $time_start")
                post_filter_conditions.append("node.timestamp <= $time_end")
            elif node_type == "Object":
                post_filter_conditions.append("node.first_seen <= $time_end")
                post_filter_conditions.append("node.last_seen >= $time_start")

        if sort_by_time:
            time_prop = nt.time_property if nt else None
            order_clause = f"ORDER BY node.{time_prop} ASC" if time_prop else "ORDER BY score DESC"
        else:
            order_clause = "ORDER BY score DESC"

        if post_filter_conditions:
            search_limit = max(limit * 50, 500) if node_type == "Keyframe" else limit * 3
        else:
            search_limit = limit
        params["search_limit"] = search_limit

        if post_filter_conditions:
            where_clause = " AND ".join(post_filter_conditions)
            query = f"""
            CALL db.index.vector.queryNodes($index_name, $search_limit, $embedding)
            YIELD node, score
            WHERE {where_clause}
            RETURN {prop_return}, score
            {order_clause}
            LIMIT $limit
            """
        else:
            query = f"""
            CALL db.index.vector.queryNodes($index_name, $search_limit, $embedding)
            YIELD node, score
            RETURN {prop_return}, score
            {order_clause}
            LIMIT $limit
            """

        try:
            records = await self._run_read(query, params)
            results = []
            for record in records:
                score = record.pop("score", 0.0)
                node_id = record.get("node_id", "")
                results.append(SearchResult(
                    node_id=node_id,
                    node_type=node_type,
                    score=score,
                    properties=record,
                ))
            return results
        except Exception as e:
            logger.error(f"HNSW vector search failed for {node_type}: {e}")
            return []

    async def _keyword_search(
        self,
        node_type: str,
        query_text: str,
        video_ids: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[SearchResult]:
        """Fulltext keyword search using Lucene indexes.

        Complements vector search by finding exact keyword matches.
        Only available for node types with fulltext indexes (Chapter, ChapterGroup).
        """
        index_name = FULLTEXT_INDEX_MAP.get(node_type)
        if not index_name:
            return []

        await self._ensure_driver()

        nt = node_registry.get(node_type)
        if not nt:
            return []

        properties = nt.neo4j_properties
        prop_return = ", ".join(f"node.{p} AS {p}" for p in properties)

        params = {
            "query_text": query_text,
            "limit": limit,
            "index_name": index_name,
        }

        video_filter = ""
        if video_ids:
            video_filter = "WHERE node.video_id IN $video_ids"
            params["video_ids"] = video_ids

        query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $query_text)
        YIELD node, score
        {video_filter}
        RETURN {prop_return}, score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            records = await self._run_read(query, params)
            results = []
            for record in records:
                score = record.pop("score", 0.0)
                node_id = record.get("node_id", "")
                results.append(SearchResult(
                    node_id=node_id,
                    node_type=node_type,
                    score=score,
                    properties=record,
                ))
            return results
        except Exception as e:
            logger.error(f"Keyword search failed for {node_type}: {e}")
            return []

    def _merge_search_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
    ) -> List[SearchResult]:
        """Merge vector and keyword search results, deduplicating by node_id.

        For duplicate nodes, keeps the one with the higher score.
        Results are sorted by score descending.
        """
        seen = {}
        for r in vector_results:
            seen[r.node_id] = r
        for r in keyword_results:
            if r.node_id not in seen or r.score > seen[r.node_id].score:
                seen[r.node_id] = r
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    # =========================================================================
    # Cross-Video Discovery
    # =========================================================================

    async def find_relevant_videos(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find relevant videos by searching ChapterGroup-level embeddings.

        Aggregates ChapterGroup results by video_id. Returns video_title in
        every result — v4 orchestrator simply ignores that field.

        Returns:
            List of dicts with video_id, max_score, top_groups, video_title.
        """
        await self._ensure_driver()

        chapter_group_type = node_registry.get("ChapterGroup")
        index_name = (
            chapter_group_type.embedding_index_name
            if chapter_group_type
            else "chaptergroup_embedding_index"
        )

        where_clause = "WHERE node.video_id IN $video_ids" if video_ids else ""

        query = f"""
        CALL db.index.vector.queryNodes($index_name, $search_limit, $embedding)
        YIELD node, score
        {where_clause}
        WITH node.video_id AS video_id,
             MAX(score) AS max_score,
             COLLECT({{summary: node.summary, name: node.name, score: score}})[0..3] AS top_groups,
             COLLECT(node.video_title)[0] AS video_title
        RETURN video_id, max_score, top_groups, video_title
        ORDER BY max_score DESC
        LIMIT $limit
        """

        params = {
            "index_name": index_name,
            "search_limit": limit * 5,
            "embedding": query_embedding,
            "limit": limit,
        }
        if video_ids:
            params["video_ids"] = video_ids

        try:
            return await self._run_read(query, params)
        except Exception as e:
            logger.error(f"Failed to find relevant videos: {e}")
            return []

    async def find_relevant_videos_by_chapter(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find relevant videos by searching Chapter-level embeddings.

        Complements ChapterGroup-based discovery at a finer granularity.
        Chapters contain detailed topic descriptions that may match queries
        missed by the coarser ChapterGroup summaries. Used by v5.

        Returns:
            List of dicts with video_id, max_score, video_title.
        """
        await self._ensure_driver()

        chapter_type = node_registry.get("Chapter")
        index_name = (
            chapter_type.embedding_index_name
            if chapter_type
            else "chapter_embedding_index"
        )

        where_clause = "WHERE node.video_id IN $video_ids" if video_ids else ""

        query = f"""
        CALL db.index.vector.queryNodes($index_name, $search_limit, $embedding)
        YIELD node, score
        {where_clause}
        WITH node.video_id AS video_id,
             MAX(score) AS max_score,
             COLLECT(node.video_title)[0] AS video_title
        RETURN video_id, max_score, video_title
        ORDER BY max_score DESC
        LIMIT $limit
        """

        params = {
            "index_name": index_name,
            "search_limit": limit * 8,
            "embedding": query_embedding,
            "limit": limit,
        }
        if video_ids:
            params["video_ids"] = video_ids

        try:
            return await self._run_read(query, params)
        except Exception as e:
            logger.error(f"Failed to find relevant videos by chapter: {e}")
            return []

    async def get_video_titles(self, video_ids: List[str]) -> Dict[str, str]:
        """Fetch video_title for a list of video IDs from ChapterGroup nodes.

        Used by v5. Returns empty string for any video_id not found.

        Returns:
            Dict mapping video_id → video_title.
        """
        if not video_ids:
            return {}
        query = """
        MATCH (cg:ChapterGroup)
        WHERE cg.video_id IN $video_ids AND cg.video_title IS NOT NULL
        RETURN DISTINCT cg.video_id AS video_id, cg.video_title AS title
        """
        try:
            records = await self._run_read(query, {"video_ids": video_ids})
            return {r["video_id"]: r["title"] for r in records}
        except Exception as e:
            logger.warning(f"Failed to fetch video titles: {e}")
            return {}

    async def aggregate_video_summary(
        self,
        video_id: str,
    ) -> Dict[str, Any]:
        """Aggregate all ChapterGroup summaries for a single video.

        Returns:
            Dict with video_id, groups (ordered list), combined_summary (str).
        """
        await self._ensure_driver()

        query = """
        MATCH (g:ChapterGroup {video_id: $video_id})
        RETURN g.node_id AS node_id,
               g.name AS name,
               g.order AS order,
               g.summary AS summary,
               g.start_time AS start_time,
               g.end_time AS end_time
        ORDER BY g.order
        """

        try:
            records = await self._run_read(query, {"video_id": video_id})

            if not records:
                return {"video_id": video_id, "groups": [], "combined_summary": ""}

            summaries = []
            for i, group in enumerate(records):
                summary = group.get("summary", "")
                name = group.get("name", f"Part {i+1}")
                if summary:
                    summaries.append(f"[{name}] {summary}")

            return {
                "video_id": video_id,
                "groups": records,
                "combined_summary": "\n\n".join(summaries),
            }
        except Exception as e:
            logger.error(f"Failed to aggregate video summary: {e}")
            return {"video_id": video_id, "groups": [], "combined_summary": ""}

    # =========================================================================
    # Utility / Catalog
    # =========================================================================

    async def get_all_video_ids(self) -> List[str]:
        """Return all unique video IDs present in the database."""
        await self._ensure_driver()

        query = """
        MATCH (n)
        WHERE n.video_id IS NOT NULL
        RETURN DISTINCT n.video_id AS video_id
        """

        try:
            records = await self._run_read(query)
            return [record["video_id"] for record in records]
        except Exception as e:
            logger.error(f"Failed to get video IDs: {e}")
            return []

    async def get_video_catalog_raw(self) -> List[Dict[str, Any]]:
        """Return a lightweight video catalog from ChapterGroup metadata.

        Each entry includes video_id, group_order, main_theme, playlist_id,
        playlist_order extracted from the metadata JSON field.
        """
        await self._ensure_driver()

        query = """
        MATCH (g:ChapterGroup)
        WHERE g.metadata IS NOT NULL
        RETURN g.video_id AS video_id,
               g.metadata AS metadata,
               g.order AS group_order
        ORDER BY g.video_id, g.order
        """

        try:
            records = await self._run_read(query)
            results = []
            for record in records:
                meta = record.get("metadata")
                if meta is None:
                    continue
                if isinstance(meta, str):
                    try:
                        meta = _json.loads(meta)
                    except _json.JSONDecodeError:
                        continue
                results.append({
                    "video_id": record["video_id"],
                    "group_order": record.get("group_order", 0),
                    "main_theme": meta.get("main_theme", ""),
                    "playlist_id": meta.get("playlist_id"),
                    "playlist_order": meta.get("playlist_order"),
                })
            return results
        except Exception as e:
            logger.error(f"Failed to get video catalog: {e}")
            return []

    async def get_node_by_id(
        self,
        node_id: str,
        node_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fetch a single node by its ID.

        Args:
            node_id: Node identifier.
            node_type: Optional node type hint for property selection via registry.

        Returns:
            Node properties dict (embedding excluded), or None if not found.
        """
        await self._ensure_driver()

        nt = node_registry.get(node_type) if node_type else None
        if nt:
            properties = nt.neo4j_properties
            prop_return = ", ".join(f"n.{p} AS {p}" for p in properties)
            query = f"""
            MATCH (n {{node_id: $node_id}})
            RETURN {prop_return}
            """
        else:
            query = """
            MATCH (n {node_id: $node_id})
            RETURN n {.*, embedding: null} AS props
            """

        try:
            records = await self._run_read(query, {"node_id": node_id})
            if not records:
                return None
            record = records[0]
            if "props" in record:
                props = dict(record["props"])
                props.pop("embedding", None)
                return props
            return dict(record)
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None

    async def get_all_nodes_for_video(
        self,
        video_id: str,
        node_type: str,
        order_by: Optional[str] = None,
        limit: int = 100,
    ) -> List[SearchResult]:
        """Fetch all nodes of a given type for a video without vector search.

        Preferred over vector search when the query requires a full overview
        of a video (e.g. "summarise this video", "list all events").
        """
        await self._ensure_driver()

        nt = node_registry.get(node_type)
        if not nt:
            logger.warning(f"Unknown node type: {node_type}")
            return []

        properties = nt.neo4j_properties
        prop_return = ", ".join(f"n.{p} AS {p}" for p in properties)

        if order_by is None:
            if node_type in ("Chapter", "Transcript"):
                order_by = "chunk_index"
            elif node_type == "ChapterGroup":
                order_by = "order"
            elif node_type == "Event":
                order_by = "timestamp"
            else:
                order_by = "node_id"

        query = f"""
        MATCH (n:{node_type} {{video_id: $video_id}})
        RETURN {prop_return}
        ORDER BY n.{order_by}
        LIMIT $limit
        """

        try:
            records = await self._run_read(query, {"video_id": video_id, "limit": limit})
            results = []
            for record in records:
                node_id = record.get("node_id", "")
                results.append(SearchResult(
                    node_id=node_id,
                    node_type=node_type,
                    score=1.0,
                    properties=record,
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to get all {node_type} for video {video_id}: {e}")
            return []

    async def check_indexes_exist(self) -> Dict[str, bool]:
        """Check which HNSW vector indexes exist in the database."""
        await self._ensure_driver()

        query = "SHOW INDEXES YIELD name RETURN name"
        expected_indexes = node_registry.build_index_map()

        try:
            records = await self._run_read(query)
            existing = {record["name"] for record in records}
            return {name: name in existing for name in expected_indexes.values()}
        except Exception as e:
            logger.error(f"Failed to check indexes: {e}")
            return {name: False for name in expected_indexes.values()}

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    # Relationship mapping: (source_type, target_type) -> (relationship, direction)
    # direction: "out" = source-[rel]->target  |  "in" = source<-[rel]-target
    #
    # Graph uses single-direction edges for hierarchy:
    #   HAS_CHAPTER: ChapterGroup → Chapter
    #   HAS_EVENT:   Chapter → Event
    #   HAS_KEYFRAME: Chapter → Keyframe
    #   HAS_TRANSCRIPT: Chapter → Transcript
    #   CONTAINS:    Event → Object
    TRAVERSAL_MAP = {
        # DOWN traversals (parent -> child)
        ("ChapterGroup", "Chapter"): ("HAS_CHAPTER", "out"),
        ("Chapter", "Event"): ("HAS_EVENT", "out"),
        ("Chapter", "Keyframe"): ("HAS_KEYFRAME", "out"),
        ("Chapter", "Transcript"): ("HAS_TRANSCRIPT", "out"),
        ("Event", "Object"): ("CONTAINS", "out"),
        # UP traversals (child -> parent)
        ("Chapter", "ChapterGroup"): ("HAS_CHAPTER", "in"),
        ("Event", "Chapter"): ("HAS_EVENT", "in"),
        ("Keyframe", "Chapter"): ("HAS_KEYFRAME", "in"),
        ("Transcript", "Chapter"): ("HAS_TRANSCRIPT", "in"),
        ("Object", "Event"): ("CONTAINS", "in"),
    }

    # Multi-hop paths for indirect relationships (e.g. Event and Keyframe are
    # siblings under Chapter, so Event → Keyframe requires Event → Chapter → Keyframe)
    MULTI_HOP_PATHS = {
        ("Event", "Keyframe"): ["Chapter"],
        ("Keyframe", "Event"): ["Chapter"],
        ("Object", "Keyframe"): ["Event", "Chapter"],
        ("Keyframe", "Object"): ["Chapter", "Event"],
        ("Event", "Transcript"): ["Chapter"],
        ("Transcript", "Event"): ["Chapter"],
    }

    def _infer_node_type(self, node_id: str) -> Optional[str]:
        """Infer node type from node_id prefix using registry."""
        return node_registry.infer_type_from_id(node_id)

    async def traverse_relationships(
        self,
        source_ids: List[str],
        target_type: str,
        source_type: Optional[str] = None,
        video_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 20,
    ) -> List[SearchResult]:
        """Traverse graph relationships up or down the node hierarchy.

        Automatically determines traversal direction from the source/target type
        pair. Supports multi-hop paths for indirect relationships (e.g. Event →
        Keyframe via Chapter).
        """
        if not source_ids:
            return []

        if source_type is None:
            source_type = self._infer_node_type(source_ids[0])
            if source_type is None:
                logger.error(f"Cannot infer source type from node_id: {source_ids[0]}")
                return []

        target_nt = node_registry.get(target_type)
        if not target_nt:
            logger.error(f"Invalid target type: {target_type}")
            return []

        traversal_key = (source_type, target_type)

        if traversal_key in self.MULTI_HOP_PATHS:
            return await self._multi_hop_traverse(
                source_ids=source_ids,
                source_type=source_type,
                target_type=target_type,
                intermediate_types=self.MULTI_HOP_PATHS[traversal_key],
                video_id=video_id,
                time_range=time_range,
                limit=limit,
            )

        if traversal_key not in self.TRAVERSAL_MAP:
            logger.error(f"No traversal path from {source_type} to {target_type}")
            return []

        relationship, direction = self.TRAVERSAL_MAP[traversal_key]

        await self._ensure_driver()

        properties = target_nt.neo4j_properties
        prop_return = ", ".join(f"t.{p} AS {p}" for p in properties)

        if direction == "out":
            match_pattern = f"(s:{source_type})-[:{relationship}]->(t:{target_type})"
        else:
            match_pattern = f"(s:{source_type})<-[:{relationship}]-(t:{target_type})"

        where_clauses = ["s.node_id IN $source_ids"]
        params: Dict[str, Any] = {"source_ids": source_ids, "limit": limit}

        if video_id:
            where_clauses.append("t.video_id = $video_id")
            params["video_id"] = video_id

        if time_range:
            start_time, end_time = time_range
            if target_type in ("Event", "Keyframe"):
                where_clauses.append("t.timestamp >= $start_time AND t.timestamp <= $end_time")
            elif target_type in ("Chapter", "ChapterGroup", "Transcript"):
                where_clauses.append("t.start_time <= $end_time AND t.end_time >= $start_time")
            params["start_time"] = start_time
            params["end_time"] = end_time

        where_clause = " AND ".join(where_clauses)

        order_by = "t.node_id"
        if target_type in ("Event", "Keyframe"):
            order_by = "t.timestamp"
        elif target_type in ("Chapter", "ChapterGroup", "Transcript"):
            order_by = "t.start_time"

        query = f"""
        MATCH {match_pattern}
        WHERE {where_clause}
        RETURN DISTINCT {prop_return}
        ORDER BY {order_by}
        LIMIT $limit
        """

        try:
            records = await self._run_read(query, params)
            results = []
            for record in records:
                node_id = record.get("node_id", "")
                results.append(SearchResult(
                    node_id=node_id,
                    node_type=target_type,
                    score=1.0,
                    properties=record,
                ))
            return results
        except Exception as e:
            logger.error(f"Traversal from {source_type} to {target_type} failed: {e}")
            return []

    async def _multi_hop_traverse(
        self,
        source_ids: List[str],
        source_type: str,
        target_type: str,
        intermediate_types: List[str],
        video_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 20,
    ) -> List[SearchResult]:
        """Execute a multi-hop traversal through intermediate node types.

        For example, Event → Keyframe requires: Event → Chapter → Keyframe.
        Time range filtering is only applied on the final hop.
        """
        current_ids = source_ids
        current_type = source_type

        for intermediate_type in intermediate_types:
            intermediate_results = await self.traverse_relationships(
                source_ids=current_ids,
                target_type=intermediate_type,
                source_type=current_type,
                video_id=video_id,
                time_range=None,  # Don't filter intermediates by time
                limit=limit * 3,
            )

            if not intermediate_results:
                logger.warning(f"Multi-hop traversal failed at {current_type} → {intermediate_type}")
                return []

            current_ids = [r.node_id for r in intermediate_results]
            current_type = intermediate_type

        return await self.traverse_relationships(
            source_ids=current_ids,
            target_type=target_type,
            source_type=current_type,
            video_id=video_id,
            time_range=time_range,
            limit=limit,
        )

    async def get_sibling_chapters(
        self,
        chapter_ids: List[str],
        limit: int = 20,
    ) -> List[SearchResult]:
        """Get all Chapter nodes that share a parent ChapterGroup.

        Given one or more Chapter node IDs, finds their parent ChapterGroup(s)
        and returns all sibling chapters ordered by chunk_index. Used by v5
        for context expansion around retrieved chapters.
        """
        if not chapter_ids:
            return []

        await self._ensure_driver()

        chapter_nt = node_registry.get("Chapter")
        properties = chapter_nt.neo4j_properties if chapter_nt else ["node_id", "video_id", "summary"]
        prop_return = ", ".join(f"sibling.{p} AS {p}" for p in properties)

        query = f"""
        MATCH (c:Chapter)<-[:HAS_CHAPTER]-(cg:ChapterGroup)-[:HAS_CHAPTER]->(sibling:Chapter)
        WHERE c.node_id IN $chapter_ids
        RETURN DISTINCT {prop_return}
        ORDER BY sibling.video_id, sibling.chunk_index
        LIMIT $limit
        """

        try:
            records = await self._run_read(query, {"chapter_ids": chapter_ids, "limit": limit})
            results = []
            for r in records:
                props = {p: r.get(p) for p in properties}
                results.append(SearchResult(
                    node_id=r["node_id"],
                    node_type="Chapter",
                    score=0.0,
                    properties=props,
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to get sibling chapters: {e}")
            return []
