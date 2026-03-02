"""Neo4j graph database provider for Azure/cloud deployments."""

import asyncio
from typing import Dict, Any, List, Optional

from loguru import logger
from mmct.providers.base.graph_db_provider import BaseGraphDBProvider


class Neo4jGraphProvider(BaseGraphDBProvider):
    """Neo4j graph database provider.
    
    Provides a production-ready graph database implementation using Neo4j,
    suitable for Azure deployments and cloud environments.
    """
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database_name: Optional[str] = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60
    ):
        """Initialize the Neo4j graph provider.
        
        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687').
            username: Neo4j username.
            password: Neo4j password.
            database_name: Name of the database to use.
            max_connection_lifetime: Maximum lifetime of a connection in seconds.
            max_connection_pool_size: Maximum number of connections in the pool.
            connection_acquisition_timeout: Timeout for acquiring a connection.
        """
        super().__init__(database_name)
        self.uri = uri
        self.username = username
        self.password = password
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout
        
        self._driver = None
        self._init_lock = asyncio.Lock()
    
    async def _ensure_driver(self) -> None:
        """Ensure Neo4j async driver is initialized (double-checked locking)."""
        if self._driver is not None:
            return
        async with self._init_lock:
            if self._driver is not None:
                return
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=self.max_connection_lifetime,
                    max_connection_pool_size=self.max_connection_pool_size,
                    connection_acquisition_timeout=self.connection_acquisition_timeout,
                )
                logger.info(
                    f"Neo4j async driver initialized for {self.uri} "
                    f"(pool_size={self.max_connection_pool_size})"
                )
            except ImportError as e:
                logger.error("neo4j driver not installed. Install with: pip install neo4j")
                raise ImportError(
                    "neo4j package is required. Install with: pip install neo4j"
                ) from e
    
    async def _run_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a Cypher read query using the async driver and connection pool."""
        await self._ensure_driver()
        async with self._driver.session(database=self.database_name) as session:
            result = await session.run(query, parameters or {})
            return [dict(record) async for record in result]
    
    async def _run_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Run a write transaction using the async driver and connection pool."""
        await self._ensure_driver()
        
        async def _tx_work(tx):
            result = await tx.run(query, parameters or {})
            return [dict(record) async for record in result]
        
        async with self._driver.session(database=self.database_name) as session:
            return await session.execute_write(_tx_work)
    
    async def create_node(
        self,
        node_id: str,
        node_type: str,
        properties: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new node in the graph."""
        props = dict(properties)
        props["id"] = node_id
        
        query = f"""
        MERGE (n:{node_type} {{id: $node_id}})
        SET n += $properties
        RETURN n
        """
        
        params = {"node_id": node_id, "properties": props}
        
        await self._run_write(query, params)
        
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
        props = properties or {}
        
        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:{edge_type}]->(b)
        SET r += $properties
        RETURN r
        """
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "properties": props
        }
        
        await self._run_write(query, params)
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "properties": props
        }
    
    async def get_node(
        self,
        node_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a node by its ID."""
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) as labels
        """
        
        result = await self._run_query(query, {"node_id": node_id})
        
        if not result:
            return None
        
        record = result[0]
        node = dict(record["n"])
        labels = record["labels"]
        node_type = labels[0] if labels else "Node"
        
        node_id_val = node.pop("id", node_id)
        
        return {
            "id": node_id_val,
            "type": node_type,
            "properties": node
        }
    
    async def delete_node(
        self,
        node_id: str,
        **kwargs
    ) -> bool:
        """Delete a node and its associated edges."""
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        result = await self._run_write(query, {"node_id": node_id})
        
        return len(result) > 0
    
    async def delete_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Delete an edge between two nodes."""
        if edge_type:
            query = f"""
            MATCH (a {{id: $source_id}})-[r:{edge_type}]->(b {{id: $target_id}})
            DELETE r
            RETURN count(r) as deleted
            """
        else:
            query = """
            MATCH (a {id: $source_id})-[r]->(b {id: $target_id})
            DELETE r
            RETURN count(r) as deleted
            """
        
        params = {"source_id": source_id, "target_id": target_id}
        
        result = await self._run_write(query, params)
        
        return len(result) > 0
    
    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes connected by edges."""
        edge_pattern = f":{edge_type}" if edge_type else ""
        
        if direction == "out":
            query = f"""
            MATCH (n {{id: $node_id}})-[r{edge_pattern}]->(m)
            RETURN m, type(r) as edge_type, properties(r) as edge_props, labels(m) as labels, 'out' as direction
            LIMIT $limit
            """
        elif direction == "in":
            query = f"""
            MATCH (n {{id: $node_id}})<-[r{edge_pattern}]-(m)
            RETURN m, type(r) as edge_type, properties(r) as edge_props, labels(m) as labels, 'in' as direction
            LIMIT $limit
            """
        else:
            query = f"""
            MATCH (n {{id: $node_id}})-[r{edge_pattern}]-(m)
            RETURN m, type(r) as edge_type, properties(r) as edge_props, labels(m) as labels,
                   CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END as direction
            LIMIT $limit
            """
        
        result = await self._run_query(query, {"node_id": node_id, "limit": limit})
        
        neighbors = []
        for record in result:
            node_props = dict(record["m"])
            neighbor_id = node_props.pop("id", None)
            labels = record["labels"]
            node_type = labels[0] if labels else "Node"
            
            neighbors.append({
                "node": {
                    "id": neighbor_id,
                    "type": node_type,
                    "properties": node_props
                },
                "edge": {
                    "type": record["edge_type"],
                    "direction": record["direction"],
                    "properties": record["edge_props"] or {}
                }
            })
        
        return neighbors
    
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Update properties of an existing node."""
        query = """
        MATCH (n {id: $node_id})
        SET n += $properties
        RETURN n, labels(n) as labels
        """
        
        result = await self._run_write(query, {"node_id": node_id, "properties": properties})
        
        if not result:
            return None
        
        record = result[0]
        node = dict(record["n"])
        labels = record["labels"]
        node_type = labels[0] if labels else "Node"
        node.pop("id", None)
        
        return {
            "id": node_id,
            "type": node_type,
            "properties": node
        }
    
    async def query(
        self,
        query_string: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a native Cypher query."""
        return await self._run_query(query_string, parameters)
    
    async def get_nodes_by_type(
        self,
        node_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        where_clause = ""
        params: Dict[str, Any] = {"limit": limit}
        
        if filters:
            conditions = []
            for i, (key, value) in enumerate(filters.items()):
                param_name = f"filter_{i}"
                conditions.append(f"n.{key} = ${param_name}")
                params[param_name] = value
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
        MATCH (n:{node_type})
        {where_clause}
        RETURN n
        LIMIT $limit
        """
        
        result = await self._run_query(query, params)
        
        nodes = []
        for record in result:
            node_props = dict(record["n"])
            nid = node_props.pop("id", None)
            nodes.append({
                "id": nid,
                "type": node_type,
                "properties": node_props
            })
        
        return nodes
    
    async def clear_database(self, **kwargs) -> bool:
        """Clear all nodes and edges from the database."""
        query = "MATCH (n) DETACH DELETE n"
        
        await self._run_write(query)
        
        logger.info("Neo4j database cleared")
        return True
    
    async def batch_create_nodes(
        self,
        nodes: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Optimized batch node creation using UNWIND."""
        if not nodes:
            return {"success": 0, "failed": 0}
        
        nodes_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            node_type = node.get("type", "Node")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            
            props = dict(node.get("properties", {}))
            props["id"] = node["id"]
            nodes_by_type[node_type].append(props)
        
        success_count = 0
        failed_count = 0
        
        for node_type, node_list in nodes_by_type.items():
            query = f"""
            UNWIND $nodes as nodeProps
            MERGE (n:{node_type} {{id: nodeProps.id}})
            SET n += nodeProps
            RETURN count(n) as created
            """
            
            try:
                await self._run_write(query, {"nodes": node_list})
                success_count += len(node_list)
            except Exception as e:
                logger.error(f"Batch node creation failed for type {node_type}: {e}")
                failed_count += len(node_list)
        
        return {"success": success_count, "failed": failed_count}
    
    async def batch_create_edges(
        self,
        edges: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Optimized batch edge creation using UNWIND."""
        if not edges:
            return {"success": 0, "failed": 0}
        
        edges_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            edge_type = edge.get("type", "RELATED")
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append({
                "source_id": edge["source_id"],
                "target_id": edge["target_id"],
                "properties": edge.get("properties", {})
            })
        
        success_count = 0
        failed_count = 0
        
        for edge_type, edge_list in edges_by_type.items():
            query = f"""
            UNWIND $edges as edgeData
            MATCH (a {{id: edgeData.source_id}})
            MATCH (b {{id: edgeData.target_id}})
            MERGE (a)-[r:{edge_type}]->(b)
            SET r += edgeData.properties
            RETURN count(r) as created
            """
            
            try:
                await self._run_write(query, {"edges": edge_list})
                success_count += len(edge_list)
            except Exception as e:
                logger.error(f"Batch edge creation failed for type {edge_type}: {e}")
                failed_count += len(edge_list)
        
        return {"success": success_count, "failed": failed_count}
    
    async def close(self) -> None:
        """Close the Neo4j async driver and release pooled connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info(f"Neo4j driver closed for {self.uri}")
