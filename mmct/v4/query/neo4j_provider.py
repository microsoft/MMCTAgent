"""Neo4j Query Provider for V4 pipeline.

Provides optimized query methods for the V4 query pipeline, supporting:
- HNSW vector search across all node types
- Graph traversal for keyframe and relationship retrieval
- Cross-video discovery via ChapterGroup aggregation
- Parallel query execution for low latency
- Time-based post-filtering for temporal queries

Uses the node type registry for dynamic configuration of:
- Index names
- Property lists
- Traversal maps
- Time filtering logic

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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from mmct.graph import node_registry


# Default HNSW search parameter
# Higher ef_search = better recall but higher latency
# Recommended: ef_search >= k (limit) for good results
DEFAULT_EF_SEARCH = 100


@dataclass
class SearchResult:
    """Result from a vector search operation."""
    
    node_id: str
    node_type: str
    score: float
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "score": self.score,
            **self.properties,
        }


class Neo4jQueryProvider:
    """Neo4j query provider for V4 pipeline using HNSW vector search.
    
    Provides optimized HNSW vector search and graph traversal methods
    for multi-granularity retrieval from the video knowledge graph.
    
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
        for attempt in range(2):
            try:
                async with self._driver.session(database=self._database) as session:
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
    # Vector Search Methods (1.2)
    # =========================================================================
    
    async def search_chapter_groups(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search ChapterGroup nodes by embedding similarity.
        
        Args:
            query_embedding: Query vector (384-dim for text).
            video_ids: Optional list of video IDs to filter.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.
            
        Returns:
            List of SearchResult ordered by similarity score.
        """
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
        """Search Chapter nodes by embedding similarity.
        
        Args:
            query_embedding: Query vector (384-dim for text).
            video_ids: Optional list of video IDs to filter.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.
            
        Returns:
            List of SearchResult ordered by similarity score.
        """
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
        """Search Event nodes by embedding similarity.
        
        Args:
            query_embedding: Query vector (384-dim for text).
            video_ids: Optional list of video IDs to filter.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.
            
        Returns:
            List of SearchResult ordered by similarity score.
        """
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
        """Search Object nodes by embedding similarity.
        
        Args:
            query_embedding: Query vector (384-dim for text).
            video_ids: Optional list of video IDs to filter.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.
            
        Returns:
            List of SearchResult ordered by similarity score.
        """
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
        
        Note: Keyframes use 512-dim image embeddings (QdrantCLIP).
        
        Args:
            query_embedding: Query vector (512-dim for images).
            video_ids: Optional list of video IDs to filter.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.
            
        Returns:
            List of SearchResult ordered by similarity score.
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
    ) -> Dict[str, List[SearchResult]]:
        """Search multiple node types in parallel.
        
        Args:
            query_embedding: Query vector.
            targets: List of node types to search (e.g., ["Chapter", "Event"]).
            video_ids: Optional list of video IDs to filter.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit_per_type: Maximum results per node type.
            sort_by_time: If True, sort results chronologically instead of by score.
            
        Returns:
            Dictionary mapping node type to list of SearchResults.
        """
        # Filter to searchable targets using registry
        valid_targets = [
            t for t in targets 
            if node_registry.get(t) and node_registry.get(t).is_searchable
        ]
        
        if not valid_targets:
            return {}
        
        # Execute searches in parallel
        tasks = [
            self._vector_search(
                node_type=target,
                query_embedding=query_embedding,
                video_ids=video_ids,
                time_range=time_range,
                limit=limit_per_type,
                sort_by_time=sort_by_time,
            )
            for target in valid_targets
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for target, result in zip(valid_targets, results):
            if isinstance(result, Exception):
                logger.error(f"Error searching {target}: {result}")
                output[target] = []
            else:
                output[target] = result
        
        return output
    
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
        
        Args:
            node_type: Type of node to search.
            query_embedding: Query vector.
            video_ids: Optional video ID filter (post-HNSW).
            time_range: Optional (start_time, end_time) tuple (post-HNSW).
            limit: Maximum results.
            sort_by_time: If True, sort results by time instead of score.
                Useful for temporal queries where chronological order matters.
            
        Returns:
            List of SearchResult.
        """
        await self._ensure_driver()
        
        # Get index name and properties from registry
        nt = node_registry.get(node_type)
        if not nt:
            logger.warning(f"Unknown node type: {node_type}")
            return []
        
        index_name = nt.embedding_index_name
        properties = nt.neo4j_properties
        prop_return = ", ".join(f"node.{p} AS {p}" for p in properties)
        
        # Use ef_search for HNSW - should be >= limit for good recall
        effective_ef = max(self._ef_search, limit * 2)
        
        params = {
            "index_name": index_name,
            "embedding": query_embedding,
            "limit": limit,
        }
        
        # Build post-filter WHERE conditions (applied after vector search)
        post_filter_conditions = []
        
        if video_ids:
            post_filter_conditions.append("node.video_id IN $video_ids")
            params["video_ids"] = video_ids
        
        if time_range:
            time_start, time_end = time_range
            params["time_start"] = time_start
            params["time_end"] = time_end
            
            # Time field mapping per node type
            if node_type in ("ChapterGroup", "Chapter", "Transcript"):
                # Range overlap: node overlaps with query range
                post_filter_conditions.append("node.start_time < $time_end")
                post_filter_conditions.append("node.end_time > $time_start")
            elif node_type in ("Event", "Keyframe"):
                # Point in range: timestamp within query range
                post_filter_conditions.append("node.timestamp >= $time_start")
                post_filter_conditions.append("node.timestamp <= $time_end")
            elif node_type == "Object":
                # Object appeared before end of range
                post_filter_conditions.append("node.first_seen <= $time_end")
                post_filter_conditions.append("node.last_seen >= $time_start")
        
        # Determine sort order
        if sort_by_time:
            # Get time property from node type registry
            time_prop = nt.time_property if nt else None
            if time_prop:
                order_clause = f"ORDER BY node.{time_prop} ASC"
            else:
                # Fallback to score if no time property
                order_clause = "ORDER BY score DESC"
        else:
            order_clause = "ORDER BY score DESC"
        
        # Build Cypher query - vector search first, then filter
        # Over-fetch to account for post-filtering
        search_limit = limit * 3 if post_filter_conditions else limit
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
            # No filtering - standard HNSW search
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
    
    # =========================================================================
    # Cross-Video Discovery (1.4)
    # =========================================================================
    
    async def find_relevant_videos(
        self,
        query_embedding: List[float],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find relevant videos by searching ChapterGroup summaries.
        
        Aggregates ChapterGroup results by video_id to find which videos
        are most relevant to a query.
        
        Args:
            query_embedding: Query vector (384-dim).
            limit: Maximum number of videos to return.
            
        Returns:
            List of dicts with video_id, max_score, and summary snippets.
        """
        await self._ensure_driver()
        
        # Search ChapterGroups and aggregate by video_id
        chapter_group_type = node_registry.get("ChapterGroup")
        index_name = chapter_group_type.embedding_index_name if chapter_group_type else "chaptergroup_embedding_index"
        
        query = """
        CALL db.index.vector.queryNodes($index_name, $search_limit, $embedding)
        YIELD node, score
        WITH node.video_id AS video_id, 
             MAX(score) AS max_score,
             COLLECT({summary: node.summary, name: node.name, score: score})[0..3] AS top_groups
        RETURN video_id, max_score, top_groups
        ORDER BY max_score DESC
        LIMIT $limit
        """
        
        try:
            records = await self._run_read(query, {
                "index_name": index_name,
                "search_limit": limit * 5,
                "embedding": query_embedding,
                "limit": limit,
            })
            return records
            
        except Exception as e:
            logger.error(f"Failed to find relevant videos: {e}")
            return []
    
    async def aggregate_video_summary(
        self,
        video_id: str,
    ) -> Dict[str, Any]:
        """Aggregate all ChapterGroup summaries for a video.
        
        Args:
            video_id: Video identifier.
            
        Returns:
            Dictionary with video_id, groups (ordered), and combined_summary.
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
                return {
                    "video_id": video_id,
                    "groups": [],
                    "combined_summary": "",
                }
            
            # Build combined summary
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
            return {
                "video_id": video_id,
                "groups": [],
                "combined_summary": "",
            }
    
    async def get_all_video_ids(self) -> List[str]:
        """Get all unique video IDs in the database.
        
        Returns:
            List of video IDs.
        """
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
        """Get lightweight video catalog from ChapterGroup metadata.
        
        Returns one row per ChapterGroup with video_id, main_theme,
        playlist_id, and playlist_order extracted from the metadata JSON.
        
        Returns:
            List of dicts with video_id, main_theme, playlist_id, playlist_order.
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
                import json as _json
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
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    async def get_node_by_id(
        self,
        node_id: str,
        node_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a single node by ID.
        
        Args:
            node_id: Node identifier.
            node_type: Optional node type for property selection.
            
        Returns:
            Node properties dict or None if not found.
        """
        await self._ensure_driver()
        
        # Get properties from registry
        nt = node_registry.get(node_type) if node_type else None
        if nt:
            properties = nt.neo4j_properties
            prop_return = ", ".join(f"n.{p} AS {p}" for p in properties)
            query = f"""
            MATCH (n {{node_id: $node_id}})
            RETURN {prop_return}
            """
        else:
            # Return all properties except embedding
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
        """Get ALL nodes of a type for a specific video (no vector search).
        
        Use this instead of vector search when:
        - Query asks for video overview/summary
        - Query needs understanding of whole video timeline
        - Query asks "what is this video about"
        
        Args:
            video_id: Video identifier.
            node_type: Node type (ChapterGroup, Chapter, Transcript, Event, Object).
            order_by: Optional property to order by (e.g., "chunk_index", "start_time").
            limit: Maximum results (default 100).
            
        Returns:
            List of SearchResult with all matching nodes.
        """
        await self._ensure_driver()
        
        # Get properties from registry
        nt = node_registry.get(node_type)
        if not nt:
            logger.warning(f"Unknown node type: {node_type}")
            return []
        
        properties = nt.neo4j_properties
        prop_return = ", ".join(f"n.{p} AS {p}" for p in properties)
        
        # Determine order field
        if order_by is None:
            # Default ordering by type
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
                    score=1.0,  # No similarity score for direct fetch
                    properties=record,
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get all {node_type} for video {video_id}: {e}")
            return []
    
    async def check_indexes_exist(self) -> Dict[str, bool]:
        """Check which vector indexes exist in the database.
        
        Returns:
            Dictionary mapping index name to existence boolean.
        """
        await self._ensure_driver()
        
        query = "SHOW INDEXES YIELD name RETURN name"
        
        # Build expected index names from registry
        expected_indexes = node_registry.build_index_map()
        
        try:
            records = await self._run_read(query)
            existing = {record["name"] for record in records}
            return {
                name: name in existing
                for name in expected_indexes.values()
            }
        except Exception as e:
            logger.error(f"Failed to check indexes: {e}")
            return {name: False for name in expected_indexes.values()}
    
    # =========================================================================
    # Unified Graph Traversal (Up/Down hierarchy)
    # =========================================================================
    
    # Relationship mapping: (source_type, target_type) -> (relationship, direction)
    # direction: "out" means traverse source-[rel]->target
    #            "in" means traverse source<-[rel]-target (reverse direction)
    # 
    # Graph uses single-direction edges for hierarchy:
    # - HAS_CHAPTER: ChapterGroup → Chapter
    # - HAS_EVENT: Chapter → Event  
    # - HAS_KEYFRAME: Chapter → Keyframe
    # - HAS_TRANSCRIPT: Chapter → Transcript
    # - CONTAINS: Event → Object
    #
    # Query code handles bidirectional traversal using direction flag.
    TRAVERSAL_MAP = {
        # DOWN traversals (parent -> child) - follow edge direction
        ("ChapterGroup", "Chapter"): ("HAS_CHAPTER", "out"),
        ("Chapter", "Event"): ("HAS_EVENT", "out"),
        ("Chapter", "Keyframe"): ("HAS_KEYFRAME", "out"),
        ("Chapter", "Transcript"): ("HAS_TRANSCRIPT", "out"),
        ("Event", "Object"): ("CONTAINS", "out"),
        # UP traversals (child -> parent) - reverse edge direction
        ("Chapter", "ChapterGroup"): ("HAS_CHAPTER", "in"),
        ("Event", "Chapter"): ("HAS_EVENT", "in"),
        ("Keyframe", "Chapter"): ("HAS_KEYFRAME", "in"),
        ("Transcript", "Chapter"): ("HAS_TRANSCRIPT", "in"),
        ("Object", "Event"): ("CONTAINS", "in"),
    }
    
    # Multi-hop traversal paths for indirect relationships
    # (source_type, target_type) -> list of intermediate types to traverse through
    # Event and Keyframe are siblings under Chapter, so we need to go through Chapter
    MULTI_HOP_PATHS = {
        ("Event", "Keyframe"): ["Chapter"],      # Event → Chapter → Keyframe
        ("Keyframe", "Event"): ["Chapter"],      # Keyframe → Chapter → Event
        ("Object", "Keyframe"): ["Event", "Chapter"],  # Object → Event → Chapter → Keyframe
        ("Keyframe", "Object"): ["Chapter", "Event"],  # Keyframe → Chapter → Event → Object
        ("Event", "Transcript"): ["Chapter"],    # Event → Chapter → Transcript
        ("Transcript", "Event"): ["Chapter"],    # Transcript → Chapter → Event
    }
    
    def _infer_node_type(self, node_id: str) -> Optional[str]:
        """Infer node type from node_id prefix using registry.
        
        Args:
            node_id: Node identifier.
            
        Returns:
            Node type string or None if cannot infer.
        """
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
        """Traverse graph relationships up or down the hierarchy.
        
        Automatically determines traversal direction based on source and target types.
        Supports filtering by video_id and time_range.
        Handles multi-hop traversals for indirect relationships (e.g., Event → Keyframe).
        
        Args:
            source_ids: List of source node IDs to traverse from.
            target_type: Target node type (ChapterGroup, Chapter, Event, Object, Keyframe).
            source_type: Optional source type (auto-detected from node_id if not provided).
            video_id: Optional video_id filter.
            time_range: Optional (start_time, end_time) filter tuple.
            limit: Maximum results to return.
            
        Returns:
            List of SearchResult for target nodes.
        """
        if not source_ids:
            return []
        
        # Infer source type if not provided
        if source_type is None:
            source_type = self._infer_node_type(source_ids[0])
            if source_type is None:
                logger.error(f"Cannot infer source type from node_id: {source_ids[0]}")
                return []
        
        # Validate target type using registry
        target_nt = node_registry.get(target_type)
        if not target_nt:
            logger.error(f"Invalid target type: {target_type}")
            return []
        
        # Check if this is a direct traversal or needs multi-hop
        traversal_key = (source_type, target_type)
        
        if traversal_key in self.MULTI_HOP_PATHS:
            # Multi-hop traversal needed (e.g., Event → Chapter → Keyframe)
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
        
        # Build Cypher query - get properties from registry
        properties = target_nt.neo4j_properties
        prop_return = ", ".join(f"t.{p} AS {p}" for p in properties)
        
        # Build MATCH pattern based on direction
        if direction == "out":
            match_pattern = f"(s:{source_type})-[:{relationship}]->(t:{target_type})"
        else:  # direction == "in"
            match_pattern = f"(s:{source_type})<-[:{relationship}]-(t:{target_type})"
        
        # Build WHERE clauses
        where_clauses = ["s.node_id IN $source_ids"]
        params = {"source_ids": source_ids, "limit": limit}
        
        if video_id:
            where_clauses.append("t.video_id = $video_id")
            params["video_id"] = video_id
        
        if time_range:
            start_time, end_time = time_range
            # Different node types use different time properties
            if target_type in ("Event",):
                where_clauses.append("t.timestamp >= $start_time AND t.timestamp <= $end_time")
            elif target_type in ("Chapter", "ChapterGroup", "Transcript"):
                where_clauses.append("t.start_time <= $end_time AND t.end_time >= $start_time")
            elif target_type == "Keyframe":
                where_clauses.append("t.timestamp >= $start_time AND t.timestamp <= $end_time")
            params["start_time"] = start_time
            params["end_time"] = end_time
        
        where_clause = " AND ".join(where_clauses)
        
        # Determine ORDER BY based on target type
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
                    score=1.0,  # Graph traversal, not vector search
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
        
        For example, Event → Keyframe requires: Event → Chapter → Keyframe
        
        Args:
            source_ids: Starting node IDs.
            source_type: Type of source nodes.
            target_type: Final target type.
            intermediate_types: List of intermediate types to traverse through.
            video_id: Optional video_id filter.
            time_range: Optional time range filter (applied to final target).
            limit: Maximum results.
            
        Returns:
            List of SearchResult for target nodes.
        """
        current_ids = source_ids
        current_type = source_type
        
        # Traverse through each intermediate type
        for intermediate_type in intermediate_types:
            intermediate_results = await self.traverse_relationships(
                source_ids=current_ids,
                target_type=intermediate_type,
                source_type=current_type,
                video_id=video_id,
                time_range=None,  # Don't filter intermediates by time
                limit=limit * 3,  # Get more intermediates to ensure enough final results
            )
            
            if not intermediate_results:
                logger.warning(f"Multi-hop traversal failed at {current_type} → {intermediate_type}")
                return []
            
            # Extract node IDs for next hop
            current_ids = [r.node_id for r in intermediate_results]
            current_type = intermediate_type
        
        # Final hop to target type (with time_range filter if provided)
        return await self.traverse_relationships(
            source_ids=current_ids,
            target_type=target_type,
            source_type=current_type,
            video_id=video_id,
            time_range=time_range,
            limit=limit,
        )
