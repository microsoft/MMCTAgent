"""Abstract base class for graph database providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict


class BaseGraphDBProvider(ABC):
    """Abstract base class for graph database providers.
    
    Provides interface for managing nodes and edges in a graph database,
    supporting temporal knowledge graphs for video analysis.
    """
    
    def __init__(self, database_name: Optional[str] = None):
        """Initialize the graph database provider.
        
        Args:
            database_name: Optional name of the database to use.
        """
        self.database_name = database_name
    
    @abstractmethod
    async def create_node(
        self,
        node_id: str,
        node_type: str,
        properties: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new node in the graph.
        
        Args:
            node_id: Unique identifier for the node.
            node_type: Type/label of the node (e.g., 'Event', 'Object').
            properties: Dictionary of node properties.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Created node with its properties.
        """
        pass
    
    @abstractmethod
    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create an edge between two nodes.
        
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Type/label of the edge (e.g., 'PRECEDES', 'CONTAINS').
            properties: Optional dictionary of edge properties.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Created edge with its properties.
        """
        pass
    
    @abstractmethod
    async def get_node(
        self,
        node_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Node properties if found, None otherwise.
        """
        pass
    
    @abstractmethod
    async def delete_node(
        self,
        node_id: str,
        **kwargs
    ) -> bool:
        """Delete a node and its associated edges.
        
        Args:
            node_id: ID of the node to delete.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            True if deletion succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    async def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Delete an edge between two nodes.
        
        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type: Optional edge type to filter deletion.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            True if deletion succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes connected by edges.
        
        Args:
            node_id: ID of the central node.
            edge_type: Optional edge type filter.
            direction: Edge direction ('in', 'out', or 'both').
            limit: Maximum number of neighbors to return.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            List of neighboring nodes with edge information.
        """
        pass
    
    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update properties of an existing node.
        
        Args:
            node_id: ID of the node to update.
            properties: Dictionary of properties to update.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Updated node properties if successful, None otherwise.
        """
        pass
    
    @abstractmethod
    async def query(
        self,
        query_string: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a native graph query.
        
        Args:
            query_string: Native query string (Cypher for Neo4j, etc.).
            parameters: Optional query parameters.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            List of query results.
        """
        pass
    
    @abstractmethod
    async def get_nodes_by_type(
        self,
        node_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type.
        
        Args:
            node_type: Type/label of nodes to retrieve.
            filters: Optional property filters.
            limit: Maximum number of nodes to return.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            List of matching nodes.
        """
        pass
    
    async def batch_create_nodes(
        self,
        nodes: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Batch create multiple nodes.
        
        Default implementation iterates over nodes.
        Subclasses may override for optimized batch operations.
        
        Args:
            nodes: List of node definitions with 'id', 'type', and 'properties'.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dictionary with 'success' count and 'failed' count.
        """
        success_count = 0
        failed_count = 0
        
        for node in nodes:
            try:
                await self.create_node(
                    node_id=node["id"],
                    node_type=node["type"],
                    properties=node.get("properties", {}),
                    **kwargs
                )
                success_count += 1
            except Exception:
                failed_count += 1
        
        return {"success": success_count, "failed": failed_count}
    
    async def batch_create_edges(
        self,
        edges: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Batch create multiple edges.
        
        Default implementation iterates over edges.
        Subclasses may override for optimized batch operations.
        
        Args:
            edges: List of edge definitions with 'source_id', 'target_id', 
                'type', and optional 'properties'.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dictionary with 'success' count and 'failed' count.
        """
        success_count = 0
        failed_count = 0
        
        for edge in edges:
            try:
                await self.create_edge(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    edge_type=edge["type"],
                    properties=edge.get("properties"),
                    **kwargs
                )
                success_count += 1
            except Exception:
                failed_count += 1
        
        return {"success": success_count, "failed": failed_count}
    
    async def clear_database(self, **kwargs) -> bool:
        """Clear all nodes and edges from the database.
        
        Use with caution - this removes all data.
        
        Args:
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            True if operation succeeded, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement clear_database")
    
    async def close(self) -> None:
        """Close provider connections and cleanup resources.
        
        Subclasses should override to perform cleanup operations.
        """
        pass
