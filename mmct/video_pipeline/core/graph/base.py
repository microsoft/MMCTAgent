"""Base classes for graph node types.

Defines the interface that all node types must implement to integrate
with the graph construction, upload, and query pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass
class EdgeDefinition:
    """Defines a relationship between node types.
    
    Attributes:
        edge_type: Relationship type name (e.g., "HAS_TRANSCRIPT").
        target_type: Target node type name (e.g., "Transcript").
        direction: "out" for source->target, "in" for source<-target.
        properties: Optional static edge properties.
    """
    edge_type: str
    target_type: str
    direction: str = "out"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalEdgeDefinition:
    """Defines temporal (NEXT/PREV) edges for a node type.
    
    Attributes:
        next_edge: Edge type for next relationship (e.g., "NEXT_CHAPTER").
        prev_edge: Edge type for previous relationship (e.g., "PREV_CHAPTER").
        order_by: Property to order nodes by (e.g., "chunk_index").
    """
    next_edge: str
    prev_edge: str
    order_by: str


class BaseNodeType(ABC):
    """Abstract base class for all graph node types.
    
    Each node type implements this interface to define its:
    - Schema (Pydantic model)
    - Neo4j properties and index configuration
    - Embedding extraction logic
    - Result formatting for search/traversal
    - Relationship definitions (edges)
    
    Example:
        class TranscriptNodeType(BaseNodeType):
            @property
            def name(self) -> str:
                return "Transcript"
            
            @property
            def id_prefix(self) -> str:
                return "transcript_"
            
            # ... implement other required methods
    """
    
    # =========================================================================
    # IDENTITY
    # =========================================================================
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Node type name used in Neo4j labels and queries.
        
        Returns:
            Type name (e.g., "Chapter", "Transcript", "Event").
        """
        pass
    
    @property
    @abstractmethod
    def id_prefix(self) -> str:
        """Node ID prefix for type inference from node IDs.
        
        Returns:
            Prefix string (e.g., "chapter_", "transcript_").
        """
        pass
    
    # =========================================================================
    # SCHEMA
    # =========================================================================
    
    @property
    @abstractmethod
    def model_class(self) -> Type["BaseModel"]:
        """Pydantic model class for this node type.
        
        Returns:
            The model class (e.g., GraphChapter, GraphTranscript).
        """
        pass
    
    @property
    @abstractmethod
    def neo4j_properties(self) -> List[str]:
        """Properties to return in Neo4j queries.
        
        IMPORTANT: Never include 'embedding' in this list.
        
        Returns:
            List of property names to include in query results.
        """
        pass
    
    # =========================================================================
    # EMBEDDING
    # =========================================================================
    
    @property
    def embedding_dimension(self) -> int:
        """Embedding vector dimension.
        
        Returns:
            Dimension size. Default 384 (BGE-small for text).
        """
        return 384
    
    @property
    def embedding_index_name(self) -> str:
        """Neo4j vector index name for this node type.
        
        Returns:
            Index name (e.g., "chapter_embedding_index").
        """
        return f"{self.name.lower()}_embedding_index"
    
    @property
    def is_searchable(self) -> bool:
        """Whether this node type supports vector search.
        
        Returns:
            True if node has embeddings and should be searchable.
        """
        return True
    
    @abstractmethod
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        """Extract text to embed from node attributes.
        
        Args:
            attrs: Node attributes dictionary.
            
        Returns:
            Text string to generate embedding from.
        """
        pass
    
    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    
    @abstractmethod
    def create_node_properties(self, instance: "BaseModel") -> Dict[str, Any]:
        """Convert model instance to Neo4j node properties.
        
        Args:
            instance: Pydantic model instance.
            
        Returns:
            Dictionary of properties for Neo4j node creation.
        """
        pass
    
    # =========================================================================
    # QUERY FORMATTING
    # =========================================================================
    
    @abstractmethod
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        """Format node properties for search/traversal result output.
        
        Should return a subset of properties relevant for display,
        with appropriate truncation for long text fields.
        
        Args:
            props: Raw node properties from Neo4j.
            
        Returns:
            Formatted properties dictionary for API response.
        """
        pass
    
    # =========================================================================
    # TIME FILTERING
    # =========================================================================
    
    @property
    def time_property(self) -> Optional[str]:
        """Property name for time-based ordering/filtering.
        
        Returns:
            - "timestamp" for point-in-time nodes (Event, Keyframe)
            - "start_time" for range nodes (Chapter, Transcript)
            - None if not time-filterable
        """
        return None
    
    @property
    def supports_time_range_filter(self) -> bool:
        """Whether node has start_time/end_time for range overlap queries.
        
        Returns:
            True if node supports time range filtering.
        """
        return False
    
    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================
    
    def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
        """Define outgoing edges from this node type.
        
        Returns:
            Dict mapping target node type name to EdgeDefinition.
            Empty dict if no outgoing edges.
        """
        return {}
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        """Define incoming edges to this node type.
        
        Used for reverse traversal (child -> parent).
        
        Returns:
            Dict mapping source node type name to EdgeDefinition.
            Empty dict if no incoming edges to define.
        """
        return {}
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        """Define NEXT/PREV temporal edges for this node type.
        
        Returns:
            TemporalEdgeDefinition or None if no temporal ordering.
        """
        return None
    
    # =========================================================================
    # AGENT PROMPT GENERATION
    # =========================================================================
    
    @property
    def description(self) -> str:
        """Short description for agent prompts.
        
        Returns:
            Human-readable description of what this node type contains.
        """
        return f"{self.name} nodes"
    
    @property
    def use_cases(self) -> List[str]:
        """Example use cases for agent prompts.
        
        Returns:
            List of query types this node type is useful for.
        """
        return []
