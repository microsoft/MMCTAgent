"""Graph linker for creating semantic similarity relationships.

Creates SIMILAR_TO edges between semantically similar events based on
embedding similarity. Object deduplication is handled upstream in the
temporal_graph step, so SAME_AS edges are no longer needed here.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from loguru import logger

from mmct.video_pipeline.core.ingestion.models import GraphEvent
from mmct.providers.base.graph_db_provider import BaseGraphDBProvider


@dataclass
class GraphLinkResult:
    """Result of graph linking operation."""
    
    similar_edges_created: int = 0
    errors: List[str] = field(default_factory=list)


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


class GraphLinker:
    """Creates semantic similarity relationships between events.
    
    Creates SIMILAR_TO edges between events with embedding similarity
    above the configured threshold.
    """
    
    EDGE_SIMILAR_TO = "SIMILAR_TO"
    
    def __init__(
        self,
        graph_provider: BaseGraphDBProvider,
        event_similarity_threshold: float = 0.75,
        max_similar_links_per_event: int = 5,
        batch_size: int = 50,
    ):
        """Initialize GraphLinker.
        
        Args:
            graph_provider: Provider for graph database operations.
            event_similarity_threshold: Minimum cosine similarity for event links.
            max_similar_links_per_event: Maximum SIMILAR_TO edges per event.
            batch_size: Number of edges to create in each batch.
        """
        self.graph_provider = graph_provider
        self.event_similarity_threshold = event_similarity_threshold
        self.max_similar_links_per_event = max_similar_links_per_event
        self.batch_size = batch_size
    
    async def link_graph(
        self,
        events: List[GraphEvent],
        video_id: str,
    ) -> GraphLinkResult:
        """Create semantic similarity links between events.
        
        Args:
            events: List of GraphEvent instances with embedding vectors.
            video_id: Video identifier for scoping.
            
        Returns:
            GraphLinkResult with counts and any errors.
        """
        result = GraphLinkResult()
        
        logger.info(f"Linking similar events for video {video_id}: {len(events)} events")
        
        similar_count, similar_errors = await self._create_similar_edges(events, video_id)
        result.similar_edges_created = similar_count
        result.errors.extend(similar_errors)
        
        logger.info(f"Graph linking complete: {result.similar_edges_created} SIMILAR_TO edges")
        
        return result
    
    async def _create_similar_edges(
        self,
        events: List[GraphEvent],
        video_id: str,
    ) -> Tuple[int, List[str]]:
        """Create SIMILAR_TO edges between semantically similar events.
        
        Uses embedding vectors to find similar events above threshold.
        
        Args:
            events: List of GraphEvent instances with embeddings.
            video_id: Video identifier.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        # Filter events with valid embeddings
        events_with_embeddings = [
            e for e in events
            if e.id and e.embedding_vector and len(e.embedding_vector) > 0
        ]
        
        if len(events_with_embeddings) < 2:
            logger.info("Insufficient events with embeddings for similarity linking")
            return 0, []
        
        logger.info(f"Computing similarity for {len(events_with_embeddings)} events")
        
        edges_to_create = []
        
        # Track links per event to enforce maximum
        links_per_event: Dict[str, int] = {}
        
        # Compare all pairs of events
        for i, event_a in enumerate(events_with_embeddings):
            if links_per_event.get(event_a.id, 0) >= self.max_similar_links_per_event:
                continue
            
            # Collect similarity scores for this event
            similarities: List[Tuple[GraphEvent, float]] = []
            
            for j, event_b in enumerate(events_with_embeddings):
                if i >= j:
                    continue
                
                if links_per_event.get(event_b.id, 0) >= self.max_similar_links_per_event:
                    continue
                
                # Compute similarity
                similarity = _cosine_similarity(
                    event_a.embedding_vector,
                    event_b.embedding_vector,
                )
                
                if similarity >= self.event_similarity_threshold:
                    similarities.append((event_b, similarity))
            
            # Sort by similarity descending and take top links
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            remaining_slots = self.max_similar_links_per_event - links_per_event.get(event_a.id, 0)
            for event_b, similarity in similarities[:remaining_slots]:
                if links_per_event.get(event_b.id, 0) >= self.max_similar_links_per_event:
                    continue
                
                # Create bidirectional SIMILAR_TO edges
                edges_to_create.append({
                    "source_id": event_a.id,
                    "target_id": event_b.id,
                    "type": self.EDGE_SIMILAR_TO,
                    "properties": {
                        "video_id": video_id,
                        "similarity_score": round(similarity, 4),
                        "source_timestamp": event_a.timestamp,
                        "target_timestamp": event_b.timestamp,
                    },
                })
                
                edges_to_create.append({
                    "source_id": event_b.id,
                    "target_id": event_a.id,
                    "type": self.EDGE_SIMILAR_TO,
                    "properties": {
                        "video_id": video_id,
                        "similarity_score": round(similarity, 4),
                        "source_timestamp": event_b.timestamp,
                        "target_timestamp": event_a.timestamp,
                    },
                })
                
                links_per_event[event_a.id] = links_per_event.get(event_a.id, 0) + 1
                links_per_event[event_b.id] = links_per_event.get(event_b.id, 0) + 1
        
        # Batch create edges
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} SIMILAR_TO edges in batch {i // self.batch_size}"
                    )
            except Exception as e:
                errors.append(f"SIMILAR_TO edge batch error: {str(e)}")
                logger.error(f"Failed to create SIMILAR_TO edges batch: {e}")
        
        return success_count, errors
