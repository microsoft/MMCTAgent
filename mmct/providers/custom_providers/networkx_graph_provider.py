"""NetworkX-based graph database provider."""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional

from loguru import logger
from mmct.providers.base.graph_db_provider import BaseGraphDBProvider


class NetworkXGraphProvider(BaseGraphDBProvider):
    """In-memory graph database provider using NetworkX.
    
    Provides a lightweight graph database implementation for local development
    and testing. Supports persistence to JSON files.
    """
    
    def __init__(
        self,
        database_name: Optional[str] = "default",
        persist_path: Optional[str] = None
    ):
        """Initialize the NetworkX graph provider.
        
        Args:
            database_name: Name of the database/graph.
            persist_path: Optional path to persist graph to disk.
        """
        super().__init__(database_name)
        self.persist_path = persist_path
        self._graph = None
        self._lock = asyncio.Lock()
        self._ensure_graph()
    
    def _ensure_graph(self) -> None:
        """Ensure NetworkX graph is initialized."""
        if self._graph is None:
            try:
                import networkx as nx
                # Use MultiDiGraph to allow multiple edges between same node pair
                # (e.g., NEXT, SIMILAR_TO, CAUSES can all exist between two events)
                self._graph = nx.MultiDiGraph()
                
                if self.persist_path and os.path.exists(self.persist_path):
                    self._load_from_file()
                    
            except ImportError as e:
                logger.error("networkx not installed. Install with: pip install networkx")
                raise ImportError(
                    "networkx package is required. Install with: pip install networkx"
                ) from e
    
    def _load_from_file(self) -> None:
        """Load graph from JSON file."""
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            import networkx as nx
            # Use MultiDiGraph to allow multiple edges between same node pair
            self._graph = nx.MultiDiGraph()
            
            for node in data.get("nodes", []):
                node_id = node["id"]
                props = node.get("properties", {})
                props["_type"] = node.get("type", "Node")
                self._graph.add_node(node_id, **props)
            
            for edge in data.get("edges", []):
                self._graph.add_edge(
                    edge["source"],
                    edge["target"],
                    _type=edge.get("type", "RELATED"),
                    **edge.get("properties", {})
                )
            
            logger.info(f"Loaded graph from {self.persist_path}: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"Failed to load graph from file: {e}")
    
    def _save_to_file(self) -> None:
        """Save graph to JSON file."""
        if not self.persist_path:
            return
        
        try:
            nodes = []
            for node_id, attrs in self._graph.nodes(data=True):
                props = dict(attrs)
                node_type = props.pop("_type", "Node")
                nodes.append({
                    "id": node_id,
                    "type": node_type,
                    "properties": props
                })
            
            edges = []
            # MultiDiGraph returns (source, target, key, data) for edges
            for source, target, key, attrs in self._graph.edges(data=True, keys=True):
                props = dict(attrs)
                edge_type = props.pop("_type", "RELATED")
                edges.append({
                    "source": source,
                    "target": target,
                    "type": edge_type,
                    "properties": props
                })
            
            data = {"nodes": nodes, "edges": edges}
            
            # Ensure parent directory exists
            parent_dir = os.path.dirname(self.persist_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            tmp_path = self.persist_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.persist_path)
            
        except Exception as e:
            logger.error(f"Failed to save graph to file: {e}")
    
    async def create_node(
        self,
        node_id: str,
        node_type: str,
        properties: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new node in the graph."""
        async with self._lock:
            props = dict(properties)
            props["_type"] = node_type
            self._graph.add_node(node_id, **props)
            self._save_to_file()
            
            return {
                "id": node_id,
                "type": node_type,
                "properties": properties
            }
    
    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create an edge between two nodes."""
        async with self._lock:
            props = dict(properties) if properties else {}
            props["_type"] = edge_type
            self._graph.add_edge(source_id, target_id, **props)
            self._save_to_file()
            
            return {
                "source_id": source_id,
                "target_id": target_id,
                "type": edge_type,
                "properties": properties or {}
            }
    
    async def get_node(
        self,
        node_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID."""
        if node_id not in self._graph:
            return None
        
        attrs = dict(self._graph.nodes[node_id])
        node_type = attrs.pop("_type", "Node")
        
        return {
            "id": node_id,
            "type": node_type,
            "properties": attrs
        }
    
    async def delete_node(
        self,
        node_id: str,
        **kwargs
    ) -> bool:
        """Delete a node and its associated edges."""
        async with self._lock:
            if node_id not in self._graph:
                return False
            
            self._graph.remove_node(node_id)
            self._save_to_file()
            return True
    
    async def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Delete an edge between two nodes."""
        async with self._lock:
            if not self._graph.has_edge(source_id, target_id):
                return False
            
            if edge_type:
                edge_data = self._graph.edges[source_id, target_id]
                if edge_data.get("_type") != edge_type:
                    return False
            
            self._graph.remove_edge(source_id, target_id)
            self._save_to_file()
            return True
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes connected by edges."""
        if node_id not in self._graph:
            return []
        
        neighbors = []
        
        if direction in ("out", "both"):
            # MultiDiGraph out_edges returns (u, v, key, data) when keys=True, data=True
            for _, target, key, edge_data in self._graph.out_edges(node_id, data=True, keys=True):
                if edge_type and edge_data.get("_type") != edge_type:
                    continue
                
                node_attrs = dict(self._graph.nodes[target])
                node_type = node_attrs.pop("_type", "Node")
                edge_props = dict(edge_data)
                edge_t = edge_props.pop("_type", "RELATED")
                
                neighbors.append({
                    "node": {
                        "id": target,
                        "type": node_type,
                        "properties": node_attrs
                    },
                    "edge": {
                        "type": edge_t,
                        "direction": "out",
                        "properties": edge_props
                    }
                })
        
        if direction in ("in", "both"):
            # MultiDiGraph in_edges returns (u, v, key, data) when keys=True, data=True
            for source, _, key, edge_data in self._graph.in_edges(node_id, data=True, keys=True):
                if edge_type and edge_data.get("_type") != edge_type:
                    continue
                
                node_attrs = dict(self._graph.nodes[source])
                node_type = node_attrs.pop("_type", "Node")
                edge_props = dict(edge_data)
                edge_t = edge_props.pop("_type", "RELATED")
                
                neighbors.append({
                    "node": {
                        "id": source,
                        "type": node_type,
                        "properties": node_attrs
                    },
                    "edge": {
                        "type": edge_t,
                        "direction": "in",
                        "properties": edge_props
                    }
                })
        
        return neighbors[:limit]
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update properties of an existing node."""
        async with self._lock:
            if node_id not in self._graph:
                return None
            
            current_attrs = self._graph.nodes[node_id]
            for key, value in properties.items():
                current_attrs[key] = value
            
            self._save_to_file()
            
            attrs = dict(current_attrs)
            node_type = attrs.pop("_type", "Node")
            
            return {
                "id": node_id,
                "type": node_type,
                "properties": attrs
            }
    
    async def query(
        self,
        query_string: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a query (limited support - returns all nodes matching type filter)."""
        results = []
        
        node_type = None
        if parameters and "node_type" in parameters:
            node_type = parameters["node_type"]
        
        for node_id, attrs in self._graph.nodes(data=True):
            if node_type and attrs.get("_type") != node_type:
                continue
            
            props = dict(attrs)
            n_type = props.pop("_type", "Node")
            results.append({
                "id": node_id,
                "type": n_type,
                "properties": props
            })
        
        return results
    
    async def get_nodes_by_type(
        self,
        node_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        results = []
        
        for node_id, attrs in self._graph.nodes(data=True):
            if attrs.get("_type") != node_type:
                continue
            
            if filters:
                match = True
                for key, value in filters.items():
                    if attrs.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            props = dict(attrs)
            props.pop("_type", None)
            results.append({
                "id": node_id,
                "type": node_type,
                "properties": props
            })
            
            if len(results) >= limit:
                break
        
        return results
    
    async def clear_database(self, **kwargs) -> bool:
        """Clear all nodes and edges from the graph."""
        async with self._lock:
            self._graph.clear()
            self._save_to_file()
            logger.info("Graph database cleared")
            return True
    
    async def close(self) -> None:
        """Save and close the provider."""
        self._save_to_file()
        logger.info(f"NetworkXGraphProvider closed for database: {self.database_name}")
