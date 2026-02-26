"""Neo4j graph store provider for bulk graph upload.

Implements BaseGraphStoreProvider for uploading NetworkX graphs to Neo4j
with support for vector indexes and batch operations.

Features:
- Batch node/edge upload with transactions
- Vector index creation for semantic search (Neo4j 5.x+)
- Video-scoped data management

Note: Embedding generation is handled by the graph_upload step, not this provider.
This provider expects nodes to already have 'embedding' attributes when uploaded.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional

from loguru import logger
import networkx as nx

from mmct.providers.base.graph_store_provider import BaseGraphStoreProvider


def _sanitize_properties_for_neo4j(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize properties for Neo4j storage.
    
    Neo4j only accepts primitive types (str, int, float, bool) or arrays thereof.
    Complex types (dicts) are serialized to JSON strings.
    
    Args:
        properties: Dictionary of node/edge properties.
        
    Returns:
        Sanitized properties dict safe for Neo4j.
    """
    sanitized = {}
    for key, value in properties.items():
        if value is None:
            continue  # Skip None values
        elif isinstance(value, dict):
            # Serialize dicts to JSON string
            sanitized[key] = json.dumps(value)
        elif isinstance(value, list):
            # Check if list contains dicts
            if value and isinstance(value[0], dict):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = value
        else:
            sanitized[key] = value
    return sanitized


class Neo4jGraphStoreProvider(BaseGraphStoreProvider):
    """Neo4j implementation of graph store provider.
    
    Uploads hierarchical video graphs to Neo4j with:
    - Batch node/edge creation for performance
    - Vector index support for embedding search (Neo4j 5.x+)
    - Video-scoped data management
    
    Node labels: ChapterGroup, Chapter, Keyframe, Event, Object
    All nodes have video_id property for multi-video support.
    
    Note: Nodes should have 'embedding' attribute pre-populated by the upload step.
    """
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        node_batch_size: int = 100,
        edge_batch_size: int = 500,
    ):
        """Initialize Neo4j graph store provider.
        
        Args:
            uri: Neo4j connection URI (bolt://host:port).
            username: Neo4j username.
            password: Neo4j password.
            database: Database name (default "neo4j").
            node_batch_size: Nodes per transaction batch.
            edge_batch_size: Edges per transaction batch.
        """
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._node_batch_size = node_batch_size
        self._edge_batch_size = edge_batch_size
        self._driver = None
    
    def _ensure_driver(self):
        """Lazy initialize Neo4j driver."""
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self._uri,
                    auth=(self._username, self._password)
                )
                # Verify connectivity
                self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self._uri}")
            except ImportError:
                raise ImportError(
                    "neo4j package is required. Install with: pip install neo4j"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
    
    async def upload_graph(
        self,
        graph: nx.Graph,
        video_id: str,
        clear_existing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a complete NetworkX graph to Neo4j.
        
        Args:
            graph: NetworkX graph with nodes having _type and embedding attributes.
            video_id: Video identifier for namespacing.
            clear_existing: Whether to clear existing data for this video.
            
        Returns:
            Dict with upload statistics.
        """
        self._ensure_driver()
        
        stats = {
            "video_id": video_id,
            "nodes_uploaded": 0,
            "edges_uploaded": 0,
            "nodes_failed": 0,
            "edges_failed": 0,
            "node_types": {},
            "edge_types": {},
        }
        
        # Clear existing data if requested
        if clear_existing:
            await self.clear_video_graph(video_id)
        
        # Extract nodes by type
        nodes_by_type: Dict[str, List[Dict]] = {}
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("_type", "Node")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            
            # Prepare node properties (exclude internal attrs, sanitize for Neo4j)
            raw_properties = {k: v for k, v in attrs.items() if not k.startswith("_")}
            raw_properties["node_id"] = node_id
            raw_properties["video_id"] = video_id
            properties = _sanitize_properties_for_neo4j(raw_properties)
            
            nodes_by_type[node_type].append({
                "id": node_id,
                "type": node_type,
                "properties": properties
            })
        
        # Upload nodes by type in batches
        for node_type, nodes in nodes_by_type.items():
            result = await self._upload_nodes_by_type(nodes, node_type)
            stats["nodes_uploaded"] += result["success"]
            stats["nodes_failed"] += result["failed"]
            stats["node_types"][node_type] = result["success"]
        
        # Extract and upload edges
        edges_by_type: Dict[str, List[Dict]] = {}
        try:
            # Handle MultiDiGraph (has keys)
            edge_iter = graph.edges(data=True, keys=True)
            for source, target, key, attrs in edge_iter:
                edge_type = attrs.get("_type", "RELATED")
                if edge_type not in edges_by_type:
                    edges_by_type[edge_type] = []
                
                raw_properties = {k: v for k, v in attrs.items() if not k.startswith("_")}
                properties = _sanitize_properties_for_neo4j(raw_properties)
                edges_by_type[edge_type].append({
                    "source": source,
                    "target": target,
                    "type": edge_type,
                    "properties": properties
                })
        except TypeError:
            # Handle regular DiGraph
            for source, target, attrs in graph.edges(data=True):
                edge_type = attrs.get("_type", "RELATED")
                if edge_type not in edges_by_type:
                    edges_by_type[edge_type] = []
                
                raw_properties = {k: v for k, v in attrs.items() if not k.startswith("_")}
                properties = _sanitize_properties_for_neo4j(raw_properties)
                edges_by_type[edge_type].append({
                    "source": source,
                    "target": target,
                    "type": edge_type,
                    "properties": properties
                })
        
        # Upload edges by type in batches
        for edge_type, edges in edges_by_type.items():
            result = await self._upload_edges_by_type(edges, edge_type)
            stats["edges_uploaded"] += result["success"]
            stats["edges_failed"] += result["failed"]
            stats["edge_types"][edge_type] = result["success"]
        
        logger.info(
            f"Uploaded graph for {video_id}: "
            f"{stats['nodes_uploaded']} nodes, {stats['edges_uploaded']} edges"
        )
        
        return stats
    
    async def _upload_nodes_by_type(
        self,
        nodes: List[Dict[str, Any]],
        node_type: str
    ) -> Dict[str, int]:
        """Upload nodes of a specific type in batches.
        
        Args:
            nodes: List of node dicts.
            node_type: Node label (ChapterGroup, Chapter, Event, Object).
            
        Returns:
            Dict with success/failed counts.
        """
        success = 0
        failed = 0
        
        # Process in batches
        for i in range(0, len(nodes), self._node_batch_size):
            batch = nodes[i:i + self._node_batch_size]
            try:
                await asyncio.to_thread(
                    self._create_nodes_batch_sync, batch, node_type
                )
                success += len(batch)
            except Exception as e:
                logger.error(f"Failed to upload {node_type} batch: {e}")
                failed += len(batch)
        
        return {"success": success, "failed": failed}
    
    def _create_nodes_batch_sync(
        self,
        nodes: List[Dict[str, Any]],
        node_type: str
    ) -> None:
        """Synchronously create a batch of nodes in a transaction."""
        # Build parameterized query for batch insert
        query = f"""
        UNWIND $nodes AS node
        MERGE (n:{node_type} {{node_id: node.properties.node_id, video_id: node.properties.video_id}})
        SET n += node.properties
        """
        
        with self._driver.session(database=self._database) as session:
            session.run(query, nodes=nodes)
    
    async def _upload_edges_by_type(
        self,
        edges: List[Dict[str, Any]],
        edge_type: str
    ) -> Dict[str, int]:
        """Upload edges of a specific type in batches.
        
        Args:
            edges: List of edge dicts.
            edge_type: Relationship type.
            
        Returns:
            Dict with success/failed counts.
        """
        success = 0
        failed = 0
        
        for i in range(0, len(edges), self._edge_batch_size):
            batch = edges[i:i + self._edge_batch_size]
            try:
                await asyncio.to_thread(
                    self._create_edges_batch_sync, batch, edge_type
                )
                success += len(batch)
            except Exception as e:
                logger.error(f"Failed to upload {edge_type} batch: {e}")
                failed += len(batch)
        
        return {"success": success, "failed": failed}
    
    def _create_edges_batch_sync(
        self,
        edges: List[Dict[str, Any]],
        edge_type: str
    ) -> None:
        """Synchronously create a batch of edges in a transaction."""
        # Match nodes by node_id and create relationship
        query = f"""
        UNWIND $edges AS edge
        MATCH (source {{node_id: edge.source}})
        MATCH (target {{node_id: edge.target}})
        MERGE (source)-[r:{edge_type}]->(target)
        SET r += edge.properties
        """
        
        with self._driver.session(database=self._database) as session:
            session.run(query, edges=edges)
    
    async def upload_nodes_batch(
        self,
        nodes: List[Dict[str, Any]],
        video_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a batch of nodes."""
        self._ensure_driver()
        
        # Group by type
        nodes_by_type: Dict[str, List[Dict]] = {}
        for node in nodes:
            node_type = node.get("type", "Node")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            
            props = node.get("properties", {}).copy()
            props["node_id"] = node["id"]
            props["video_id"] = video_id
            
            nodes_by_type[node_type].append({
                "id": node["id"],
                "type": node_type,
                "properties": props
            })
        
        total_success = 0
        total_failed = 0
        
        for node_type, typed_nodes in nodes_by_type.items():
            result = await self._upload_nodes_by_type(typed_nodes, node_type)
            total_success += result["success"]
            total_failed += result["failed"]
        
        return {"success": total_success, "failed": total_failed}
    
    async def upload_edges_batch(
        self,
        edges: List[Dict[str, Any]],
        video_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Upload a batch of edges."""
        self._ensure_driver()
        
        # Group by type
        edges_by_type: Dict[str, List[Dict]] = {}
        for edge in edges:
            edge_type = edge.get("type", "RELATED")
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)
        
        total_success = 0
        total_failed = 0
        
        for edge_type, typed_edges in edges_by_type.items():
            result = await self._upload_edges_by_type(typed_edges, edge_type)
            total_success += result["success"]
            total_failed += result["failed"]
        
        return {"success": total_success, "failed": total_failed}
    
    async def clear_video_graph(
        self,
        video_id: str,
        **kwargs
    ) -> bool:
        """Clear all graph data for a specific video."""
        self._ensure_driver()
        
        try:
            def clear_sync():
                query = """
                MATCH (n {video_id: $video_id})
                DETACH DELETE n
                """
                with self._driver.session(database=self._database) as session:
                    result = session.run(query, video_id=video_id)
                    summary = result.consume()
                    return summary.counters.nodes_deleted
            
            deleted = await asyncio.to_thread(clear_sync)
            logger.info(f"Cleared {deleted} nodes for video {video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear video graph: {e}")
            return False
    
    async def create_vector_indexes(
        self,
        dimension: int = 384,
        node_types: Optional[List[str]] = None,
        index_suffix: str = "",
        m: int = 16,
        ef_construction: int = 128,
        **kwargs
    ) -> bool:
        """Create HNSW vector indexes for node types.
        
        Creates HNSW (Hierarchical Navigable Small World) vector indexes
        with cosine similarity for embedding-based search.
        Requires Neo4j 5.11+ with vector index support.
        
        Args:
            dimension: Embedding dimension (default: 384 for BGE-small).
            node_types: List of node types to index (default: text node types).
            index_suffix: Suffix to add to index names (e.g., "_image").
            m: HNSW M parameter - max connections per node (default: 16).
                Higher = better recall but more memory/slower indexing.
            ef_construction: HNSW efConstruction - search depth during build (default: 128).
                Higher = better index quality but slower indexing.
        """
        self._ensure_driver()
        
        # Default to text-based node types if not specified
        if node_types is None:
            node_types = ["ChapterGroup", "Chapter", "Transcript", "Event", "Object"]
        
        try:
            def create_indexes_sync():
                with self._driver.session(database=self._database) as session:
                    for node_type in node_types:
                        index_name = f"{node_type.lower()}_embedding_index{index_suffix}"
                        try:
                            query = f"""
                            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                            FOR (n:{node_type})
                            ON (n.embedding)
                            OPTIONS {{
                                indexConfig: {{
                                    `vector.dimensions`: {dimension},
                                    `vector.similarity_function`: 'cosine',
                                    `vector.hnsw.m`: {m},
                                    `vector.hnsw.ef_construction`: {ef_construction}
                                }}
                            }}
                            """
                            session.run(query)
                            logger.info(f"Created HNSW vector index: {index_name} (dim={dimension}, m={m}, ef={ef_construction})")
                        except Exception as e:
                            # Index may already exist or Neo4j version doesn't support it
                            logger.warning(f"Could not create index {index_name}: {e}")
            
            await asyncio.to_thread(create_indexes_sync)
            return True
        except Exception as e:
            logger.error(f"Failed to create vector indexes: {e}")
            return False
    
    async def get_video_stats(
        self,
        video_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get statistics for a video's graph data."""
        self._ensure_driver()
        
        try:
            def get_stats_sync():
                stats = {"video_id": video_id, "node_counts": {}, "edge_counts": {}}
                
                with self._driver.session(database=self._database) as session:
                    # Count nodes by type (including Keyframe)
                    for node_type in ["ChapterGroup", "Chapter", "Keyframe", "Event", "Object"]:
                        result = session.run(
                            f"MATCH (n:{node_type} {{video_id: $video_id}}) RETURN count(n) as count",
                            video_id=video_id
                        )
                        record = result.single()
                        stats["node_counts"][node_type] = record["count"] if record else 0
                    
                    # Count total edges
                    result = session.run(
                        "MATCH (n {video_id: $video_id})-[r]->() RETURN type(r) as type, count(r) as count",
                        video_id=video_id
                    )
                    for record in result:
                        stats["edge_counts"][record["type"]] = record["count"]
                
                return stats
            
            return await asyncio.to_thread(get_stats_sync)
        except Exception as e:
            logger.error(f"Failed to get video stats: {e}")
            return {"video_id": video_id, "error": str(e)}
    
    async def close(self) -> None:
        """Close Neo4j driver."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
