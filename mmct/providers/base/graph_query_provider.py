"""Abstract base class for Neo4j query providers used in v4 and v5 pipelines.

Covers the full union of methods required by both pipelines:
- v4: vector search, graph traversal, cross-video discovery
- v5: all of the above + chapter-level discovery, sibling expansion, video titles
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SearchResult:
    """Result from a vector or keyword search operation."""

    node_id: str
    node_type: str
    score: float
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "score": self.score,
            **self.properties,
        }


class BaseGraphQueryProvider(ABC):
    """Abstract base for Neo4j query providers used across v4 and v5 pipelines.

    Subclasses must implement all abstract methods. Internal helpers such as
    _vector_search, _keyword_search, _merge_search_results, and _ensure_driver
    are implementation details left to concrete classes.
    """

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def check_health(self) -> Dict[str, Any]:
        """Verify that the graph database is reachable.

        Returns:
            Dict with at least ``{"status": "ok"}`` on success,
            or ``{"status": "error", "error": "..."}`` on failure.
        """
        try:
            ids = await self.get_all_video_ids()
            return {"status": "ok", "video_count": len(ids)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying driver and release pooled connections."""

    # =========================================================================
    # Granular Vector Search
    # =========================================================================

    @abstractmethod
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
            video_ids: Optional list of video IDs to filter results.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.

        Returns:
            List of SearchResult ordered by similarity score descending.
        """

    @abstractmethod
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
            video_ids: Optional list of video IDs to filter results.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.

        Returns:
            List of SearchResult ordered by similarity score descending.
        """

    @abstractmethod
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
            video_ids: Optional list of video IDs to filter results.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.

        Returns:
            List of SearchResult ordered by similarity score descending.
        """

    @abstractmethod
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
            video_ids: Optional list of video IDs to filter results.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.

        Returns:
            List of SearchResult ordered by similarity score descending.
        """

    @abstractmethod
    async def search_keyframes(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[SearchResult]:
        """Search Keyframe nodes by image embedding similarity.

        Note: Keyframes use 512-dim image embeddings (e.g. QdrantCLIP), not
        the 384-dim text embeddings used by other node types.

        Args:
            query_embedding: Query vector (512-dim for images).
            video_ids: Optional list of video IDs to filter results.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit: Maximum results to return.

        Returns:
            List of SearchResult ordered by similarity score descending.
        """

    @abstractmethod
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
        """Search multiple node types in parallel with hybrid vector + keyword search.

        Args:
            query_embedding: Query vector for HNSW search.
            targets: List of node type names to search (e.g. ["Chapter", "Event"]).
            video_ids: Optional list of video IDs to filter results.
            time_range: Optional (start_time, end_time) tuple for temporal filtering.
            limit_per_type: Maximum results per node type.
            sort_by_time: If True, sort results chronologically instead of by score.
            query_text: Optional raw query text to enable hybrid keyword search.

        Returns:
            Dict mapping node type name to list of SearchResults.
        """

    # =========================================================================
    # Cross-Video Discovery
    # =========================================================================

    @abstractmethod
    async def find_relevant_videos(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find relevant videos by searching ChapterGroup-level embeddings.

        Aggregates ChapterGroup results by video_id to surface the most relevant
        videos for a query. Used by both v4 and v5.

        Args:
            query_embedding: Query vector (384-dim).
            video_ids: Optional list of video IDs to constrain the search.
            limit: Maximum number of videos to return.

        Returns:
            List of dicts with at minimum: video_id, max_score, top_groups.
            v5 implementations also include video_title.
        """

    @abstractmethod
    async def find_relevant_videos_by_chapter(
        self,
        query_embedding: List[float],
        video_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find relevant videos by searching Chapter-level embeddings.

        Complements ChapterGroup-based discovery at a finer granularity.
        Chapters contain detailed topic descriptions that may match queries
        missed by the coarser ChapterGroup summaries. Introduced in v5.

        Args:
            query_embedding: Query vector (384-dim).
            video_ids: Optional list of video IDs to constrain the search.
            limit: Maximum number of videos to return.

        Returns:
            List of dicts with video_id, max_score, and video_title.
        """

    @abstractmethod
    async def get_video_titles(
        self,
        video_ids: List[str],
    ) -> Dict[str, str]:
        """Fetch video_title for a list of video IDs.

        Looks up titles from ChapterGroup nodes. Introduced in v5.

        Args:
            video_ids: List of video IDs to look up.

        Returns:
            Dict mapping video_id → video_title (empty string if not found).
        """

    @abstractmethod
    async def aggregate_video_summary(
        self,
        video_id: str,
    ) -> Dict[str, Any]:
        """Aggregate all ChapterGroup summaries for a single video.

        Args:
            video_id: Video identifier.

        Returns:
            Dict with keys: video_id, groups (ordered list), combined_summary (str).
        """

    # =========================================================================
    # Utility / Catalog
    # =========================================================================

    @abstractmethod
    async def get_all_video_ids(self) -> List[str]:
        """Return all unique video IDs present in the database.

        Returns:
            List of video ID strings.
        """

    @abstractmethod
    async def get_video_catalog_raw(self) -> List[Dict[str, Any]]:
        """Return a lightweight video catalog from ChapterGroup metadata.

        Each entry corresponds to one ChapterGroup and includes:
        video_id, group_order, main_theme, playlist_id, playlist_order.

        Returns:
            List of dicts, one per ChapterGroup.
        """

    @abstractmethod
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

    @abstractmethod
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

        Args:
            video_id: Video identifier.
            node_type: Node type label (e.g. "Chapter", "ChapterGroup", "Event").
            order_by: Optional property name to order by (defaults chosen per type).
            limit: Maximum number of nodes to return.

        Returns:
            List of SearchResult with score=1.0 (no similarity ranking).
        """

    @abstractmethod
    async def check_indexes_exist(self) -> Dict[str, bool]:
        """Check which HNSW vector indexes exist in the database.

        Returns:
            Dict mapping index name → True if it exists, False otherwise.
        """

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    @abstractmethod
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

        Automatically determines traversal direction (parent→child or child→parent)
        from the source/target type pair. Supports multi-hop paths for indirect
        relationships (e.g. Event → Keyframe via Chapter).

        Args:
            source_ids: List of source node IDs to start from.
            target_type: Target node type to traverse to.
            source_type: Optional source type hint (auto-detected from node_id if omitted).
            video_id: Optional video_id filter applied after traversal.
            time_range: Optional (start_time, end_time) filter tuple.
            limit: Maximum results to return.

        Returns:
            List of SearchResult for matched target nodes.
        """

    @abstractmethod
    async def get_sibling_chapters(
        self,
        chapter_ids: List[str],
        limit: int = 20,
    ) -> List[SearchResult]:
        """Get all Chapter nodes that share a parent ChapterGroup.

        Given one or more Chapter node IDs, finds their parent ChapterGroup(s)
        and returns all sibling chapters in those groups ordered by chunk_index.
        Used by v5 for context expansion around retrieved chapters.

        Args:
            chapter_ids: List of Chapter node IDs.
            limit: Maximum sibling chapters to return.

        Returns:
            List of SearchResult for sibling Chapter nodes, score=0.0.
        """
