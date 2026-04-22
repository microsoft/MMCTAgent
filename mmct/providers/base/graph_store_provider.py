"""Abstract base class for graph store providers.

Provides interface for bulk graph upload/sync operations to external graph databases.
This is different from BaseGraphDBProvider which is for CRUD operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from mmct.providers.base.database_context import get_database_override

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None


class BaseGraphStoreProvider(ABC):
    """Abstract base class for graph store providers.
    
    Provides interface for bulk upload and sync operations to external
    graph databases like Neo4j, Neptune, etc.
    
    Key differences from BaseGraphDBProvider:
    - Optimized for bulk operations, not individual CRUD
    - Designed for uploading complete graphs from NetworkX
    - Supports vector embeddings as node properties
    """

    def get_database(self, default: Optional[str] = None) -> Optional[str]:
        """Return the active database name for the current request.

        Checks the per-request ``database_override`` context variable first;
        falls back to *default*.
        """
        return get_database_override() or default
    
    @abstractmethod
    async def upload_graph(
        self,
        graph: nx.Graph,
        video_id: str,
        clear_existing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a complete NetworkX graph to the store.
        
        Args:
            graph: NetworkX graph to upload (nodes should have _type attribute).
            video_id: Video identifier for namespacing.
            clear_existing: Whether to clear existing data for this video.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dict with upload statistics (nodes_uploaded, edges_uploaded, etc).
        """
        pass
    
    @abstractmethod
    async def upload_nodes_batch(
        self,
        nodes: List[Dict[str, Any]],
        video_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a batch of nodes.
        
        Args:
            nodes: List of node dicts with 'id', 'type', 'properties'.
            video_id: Video identifier.
            **kwargs: Additional parameters.
            
        Returns:
            Dict with success/failed counts.
        """
        pass
    
    @abstractmethod
    async def upload_edges_batch(
        self,
        edges: List[Dict[str, Any]],
        video_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a batch of edges.
        
        Args:
            edges: List of edge dicts with 'source', 'target', 'type', 'properties'.
            video_id: Video identifier.
            **kwargs: Additional parameters.
            
        Returns:
            Dict with success/failed counts.
        """
        pass
    
    @abstractmethod
    async def clear_video_graph(
        self,
        video_id: str,
        **kwargs
    ) -> bool:
        """Clear all graph data for a specific video.
        
        Args:
            video_id: Video identifier to clear.
            **kwargs: Additional parameters.
            
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    async def create_vector_indexes(
        self,
        dimension: int = 384,
        **kwargs
    ) -> bool:
        """Create vector indexes for embedding-based search.
        
        Args:
            dimension: Embedding dimension (default 384 for BGE-small).
            **kwargs: Additional parameters.
            
        Returns:
            True if indexes created successfully.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        pass
    
    async def get_video_stats(
        self,
        video_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get statistics for a video's graph data.
        
        Args:
            video_id: Video identifier.
            **kwargs: Additional parameters.
            
        Returns:
            Dict with node/edge counts by type.
        """
        raise NotImplementedError("Subclasses may implement get_video_stats")
