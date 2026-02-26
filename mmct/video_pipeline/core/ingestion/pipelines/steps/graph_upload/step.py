"""Graph upload pipeline step.

Uploads the hierarchical graph to a graph store with embeddings.
Embedding generation happens in this step, then upload is delegated to the provider.

Supported providers:
- Neo4jGraphStoreProvider: Upload to Neo4j with vector indexes

This step:
1. Reads the NetworkX graph from graph_construction step
2. Generates text embeddings (BGE-small, 384-dim) for text nodes
3. Generates image embeddings (QdrantCLIP, 512-dim) for Keyframe nodes
4. Uploads the embedded graph via the configured provider
"""

import time
from typing import Dict, Any, Optional, List, Tuple

from loguru import logger
import networkx as nx

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.providers.base.graph_store_provider import BaseGraphStoreProvider


# Embedding dimensions
TEXT_EMBEDDING_DIM = 384  # BGE-small
IMAGE_EMBEDDING_DIM = 512  # QdrantCLIP


@register_step("ingestion.graph_upload")
class GraphUploadStep(PipelineStep):
    """Pipeline step for uploading graph to a graph store with embeddings.
    
    Generates embeddings in this step:
    - Text nodes (ChapterGroup, Chapter, Transcript, Event, Object): BGE-small (384-dim)
    - Image nodes (Keyframe): QdrantCLIP (512-dim)
    
    Then uploads the embedded graph via the configured graph store provider.
    
    Params:
        graph_step: Source step for NetworkX graph (default: "graph_construction")
        provider: Graph store provider type: "neo4j" (default: "neo4j")
        clear_existing: Clear existing video data before upload (default: True)
        create_indexes: Create vector indexes for search (default: True)
        embedding_batch_size: Batch size for embedding generation (default: 32)
        
        # Provider-specific params (passed to provider constructor):
        # Neo4j:
        #   neo4j_uri, neo4j_username, neo4j_password, neo4j_database
        #   node_batch_size, edge_batch_size
    """
    
    step_type = "ingestion.graph_upload"
    description = "Generate embeddings and upload graph to graph store."
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute graph upload with embedding generation.
        
        Args:
            context: Pipeline step context with data store and providers.
            
        Returns:
            StepResult with upload statistics.
        """
        # Get source step
        graph_step: str = self.get_param(
            "graph_step", context, default="graph_construction"
        )
        
        # Get graph provider from graph construction step
        graph_provider = context.data_store.get(graph_step, "graph_provider")
        
        if not graph_provider or not hasattr(graph_provider, '_graph'):
            logger.error("No graph provider found in graph_construction step")
            return StepResult(
                step_id=self.step_id,
                outputs={"upload_complete": False, "error": "No graph provider found"},
                metrics={"nodes_uploaded": 0, "edges_uploaded": 0},
            )
        
        nx_graph = graph_provider._graph
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, nothing to upload")
            return StepResult(
                step_id=self.step_id,
                outputs={"upload_complete": False, "error": "Graph is empty"},
                metrics={"nodes_uploaded": 0, "edges_uploaded": 0},
            )
        
        video_id: str = getattr(context, "video_id", "unknown")
        logger.info(
            f"Starting graph upload for video {video_id}: "
            f"{nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges"
        )
        
        # Get options
        clear_existing: bool = self.get_param("clear_existing", context, default=True)
        create_indexes: bool = self.get_param("create_indexes", context, default=True)
        embedding_batch_size: int = self.get_param("embedding_batch_size", context, default=32)
        
        stats = {
            "video_id": video_id,
            "total_nodes": nx_graph.number_of_nodes(),
            "total_edges": nx_graph.number_of_edges(),
            "text_nodes_embedded": 0,
            "keyframes_embedded": 0,
            "nodes_uploaded": 0,
            "edges_uploaded": 0,
            "text_embedding_time_seconds": 0,
            "image_embedding_time_seconds": 0,
            "upload_time_seconds": 0,
        }
        
        # Step 1: Generate embeddings for all nodes
        embed_stats = await self._generate_embeddings(
            nx_graph, embedding_batch_size
        )
        stats.update(embed_stats)
        
        # Step 2: Create graph store provider and upload
        store_provider = self._create_store_provider(context)
        
        try:
            # Create vector indexes if needed
            if create_indexes:
                logger.info("Creating vector indexes...")
                await store_provider.create_vector_indexes(dimension=TEXT_EMBEDDING_DIM)
                # Create separate index for keyframes (512-dim)
                if stats["keyframes_embedded"] > 0:
                    await store_provider.create_vector_indexes(
                        dimension=IMAGE_EMBEDDING_DIM,
                        node_types=["Keyframe"],
                        index_suffix="_image"
                    )
            
            # Upload the graph (embeddings already attached to nodes)
            logger.info("Uploading graph...")
            upload_start = time.time()
            
            upload_result = await store_provider.upload_graph(
                graph=nx_graph,
                video_id=video_id,
                clear_existing=clear_existing,
            )
            
            stats["upload_time_seconds"] = round(time.time() - upload_start, 2)
            stats["nodes_uploaded"] = upload_result.get("nodes_uploaded", 0)
            stats["edges_uploaded"] = upload_result.get("edges_uploaded", 0)
            stats["node_types"] = upload_result.get("node_types", {})
            stats["edge_types"] = upload_result.get("edge_types", {})
            
            logger.info(
                f"Graph upload complete: {stats['nodes_uploaded']} nodes, "
                f"{stats['edges_uploaded']} edges "
                f"(text embeddings: {stats['text_nodes_embedded']}, "
                f"image embeddings: {stats['keyframes_embedded']})"
            )
            
            return StepResult(
                step_id=self.step_id,
                outputs={
                    "upload_complete": True,
                    "video_id": video_id,
                    "nodes_uploaded": stats["nodes_uploaded"],
                    "edges_uploaded": stats["edges_uploaded"],
                    "text_nodes_embedded": stats["text_nodes_embedded"],
                    "keyframes_embedded": stats["keyframes_embedded"],
                    "node_types": stats.get("node_types", {}),
                    "edge_types": stats.get("edge_types", {}),
                },
                metrics=stats,
            )
            
        except Exception as e:
            logger.error(f"Graph upload failed: {e}")
            return StepResult(
                step_id=self.step_id,
                outputs={"upload_complete": False, "error": str(e)},
                metrics={"nodes_uploaded": 0, "edges_uploaded": 0},
            )
        finally:
            await store_provider.close()
    
    async def _generate_embeddings(
        self,
        graph: nx.Graph,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Generate embeddings for all nodes in the graph.
        
        Text nodes (ChapterGroup, Chapter, Transcript, Event, Object) use BGE-small (384-dim).
        Image nodes (Keyframe) use QdrantCLIP (512-dim).
        
        Args:
            graph: NetworkX graph with nodes.
            batch_size: Batch size for embedding generation.
            
        Returns:
            Dict with embedding statistics.
        """
        stats = {
            "text_nodes_embedded": 0,
            "keyframes_embedded": 0,
            "text_embedding_time_seconds": 0,
            "image_embedding_time_seconds": 0,
        }
        
        # Separate nodes by type
        text_nodes: List[Tuple[str, str]] = []  # (node_id, text)
        keyframe_nodes: List[Tuple[str, str]] = []  # (node_id, filepath)
        
        for node_id, attrs in graph.nodes(data=True):
            node_type = attrs.get("_type", "Node")
            
            if node_type == "Keyframe":
                filepath = attrs.get("filepath", "")
                if filepath:
                    keyframe_nodes.append((node_id, filepath))
            else:
                text = self._get_embedding_text(node_type, attrs)
                if text:
                    text_nodes.append((node_id, text))
        
        # Generate text embeddings (BGE-small, 384-dim)
        if text_nodes:
            logger.info(f"Generating text embeddings for {len(text_nodes)} nodes...")
            text_start = time.time()
            
            from mmct.providers.custom_providers import FastEmbedBGEsmallEmbeddingProvider
            text_provider = FastEmbedBGEsmallEmbeddingProvider()
            
            for i in range(0, len(text_nodes), batch_size):
                batch = text_nodes[i:i + batch_size]
                texts = [text for _, text in batch]
                
                try:
                    embeddings = await text_provider.batch_embedding(texts)
                    for (node_id, _), embedding in zip(batch, embeddings):
                        graph.nodes[node_id]["embedding"] = embedding
                    stats["text_nodes_embedded"] += len(batch)
                except Exception as e:
                    logger.error(f"Failed to embed text batch: {e}")
            
            stats["text_embedding_time_seconds"] = round(time.time() - text_start, 2)
            logger.info(
                f"Text embeddings complete: {stats['text_nodes_embedded']} nodes "
                f"in {stats['text_embedding_time_seconds']}s"
            )
        
        # Generate image embeddings (QdrantCLIP, 512-dim)
        if keyframe_nodes:
            logger.info(f"Generating image embeddings for {len(keyframe_nodes)} keyframes...")
            img_start = time.time()
            
            try:
                from mmct.providers.custom_providers import get_qdrant_clip_provider
                image_provider = get_qdrant_clip_provider()
                
                for i in range(0, len(keyframe_nodes), batch_size):
                    batch = keyframe_nodes[i:i + batch_size]
                    filepaths = [fp for _, fp in batch]
                    
                    try:
                        embeddings = await image_provider.batch_image_embedding(filepaths)
                        for (node_id, _), embedding in zip(batch, embeddings):
                            graph.nodes[node_id]["embedding"] = embedding
                        stats["keyframes_embedded"] += len(batch)
                    except Exception as e:
                        logger.error(f"Failed to embed image batch: {e}")
                
                stats["image_embedding_time_seconds"] = round(time.time() - img_start, 2)
                logger.info(
                    f"Image embeddings complete: {stats['keyframes_embedded']} keyframes "
                    f"in {stats['image_embedding_time_seconds']}s"
                )
            except Exception as e:
                logger.error(f"Failed to initialize image embedding provider: {e}")
        
        return stats
    
    def _get_embedding_text(self, node_type: str, attrs: Dict[str, Any]) -> str:
        """Get text to embed for a node based on its type.
        
        Uses the node type registry to extract embedding text from node attributes.
        Falls back to common patterns if type not found in registry.
        
        Args:
            node_type: Node type (ChapterGroup, Chapter, Transcript, Event, Object).
            attrs: Node attributes.
            
        Returns:
            Text string to embed.
        """
        # Try registry first
        from mmct.graph import node_registry
        nt = node_registry.get(node_type)
        if nt:
            return nt.get_embedding_text(attrs)
        
        # Fallback for unknown types
        return attrs.get("description", "") or attrs.get("summary", "") or attrs.get("name", "")
    
    def _create_store_provider(self, context: StepContext) -> BaseGraphStoreProvider:
        """Create graph store provider based on configuration.
        
        Args:
            context: Pipeline step context.
            
        Returns:
            Configured BaseGraphStoreProvider instance.
        """
        provider_type: str = self.get_param("provider", context, default="neo4j")
        
        if provider_type == "neo4j":
            return self._create_neo4j_provider(context)
        else:
            raise ValueError(f"Unknown graph store provider: {provider_type}")
    
    def _create_neo4j_provider(self, context: StepContext) -> BaseGraphStoreProvider:
        """Create Neo4j graph store provider.
        
        Args:
            context: Pipeline step context.
            
        Returns:
            Configured Neo4jGraphStoreProvider instance.
        """
        from mmct.providers.custom_providers import Neo4jGraphStoreProvider
        
        # Get connection settings from params or environment
        neo4j_uri = self.get_param("neo4j_uri", context, default=None)
        neo4j_username = self.get_param("neo4j_username", context, default=None)
        neo4j_password = self.get_param("neo4j_password", context, default=None)
        neo4j_database = self.get_param("neo4j_database", context, default="neo4j")
        
        # Fall back to environment
        if not neo4j_uri or not neo4j_password:
            try:
                from app.config.provider_config import get_settings
                settings = get_settings()
                neo4j_uri = neo4j_uri or settings.neo4j_uri
                neo4j_username = neo4j_username or settings.neo4j_username
                neo4j_password = neo4j_password or settings.neo4j_password
                neo4j_database = neo4j_database or settings.neo4j_database
            except Exception as e:
                logger.warning(f"Could not load Neo4j settings from environment: {e}")
        
        # Final defaults
        neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        neo4j_username = neo4j_username or "neo4j"
        neo4j_password = neo4j_password or ""
        
        # Get batch sizes
        node_batch_size: int = self.get_param("node_batch_size", context, default=100)
        edge_batch_size: int = self.get_param("edge_batch_size", context, default=500)
        
        return Neo4jGraphStoreProvider(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
            node_batch_size=node_batch_size,
            edge_batch_size=edge_batch_size,
        )
