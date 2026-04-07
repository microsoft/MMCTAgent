"""Object node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.graph.registry import node_registry


class ObjectNodeType(BaseNodeType):
    """Object node - entities (people, items) that appear in events.
    
    Objects represent entities identified in the video that participate
    in events. Can be people, physical objects, animals, or other
    trackable entities.
    """
    
    @property
    def name(self) -> str:
        return "Object"
    
    @property
    def id_prefix(self) -> str:
        return "object_"
    
    @property
    def model_class(self) -> Type:
        from mmct.video_pipeline.core.ingestion.models import GraphObject
        return GraphObject
    
    @property
    def neo4j_properties(self) -> List[str]:
        return [
            "node_id", "video_id", "name", "first_seen", "last_seen",
            "object_type", "appearance", "identity", "is_canonical", "appearance_count"
        ]
    
    @property
    def embedding_index_name(self) -> str:
        # Objects use a different embedding model (Arctic)
        return "object_embedding_index"
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        name = attrs.get("name", "")
        attributes = attrs.get("attributes", [])
        if isinstance(attributes, list):
            attr_str = ", ".join(str(a) for a in attributes)
            return f"{name}: {attr_str}" if attr_str else name
        return name
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        props = {
            "name": instance.name or "",
            "video_id": instance.video_id or "",
            "first_seen": instance.first_seen or 0.0,
            "last_seen": instance.last_seen or 0.0,
            "object_type": instance.object_type or "unknown",
            "appearance_count": instance.appearance_count or 1,
        }
        
        # Optional fields
        if instance.appearance:
            props["appearance"] = instance.appearance
        if instance.identity:
            props["identity"] = instance.identity
        if hasattr(instance, "is_canonical"):
            props["is_canonical"] = instance.is_canonical
            
        return props
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": props.get("name", ""),
            "description": props.get("description", ""),
            "first_appearance": props.get("first_seen") or props.get("first_appearance"),
            "last_seen": props.get("last_seen"),
            "occurrences": props.get("appearance_count") or props.get("occurrences"),
            "object_type": props.get("object_type"),
        }
    
    @property
    def time_property(self) -> Optional[str]:
        return "first_seen"
    
    @property
    def supports_time_range_filter(self) -> bool:
        return False
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Event": EdgeDefinition("CONTAINS", "Event", "in"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        return None  # Objects don't have temporal ordering
    
    @property
    def description(self) -> str:
        return "Entities (people, items) that appear in events"
    
    @property
    def use_cases(self) -> List[str]:
        return [
            "Object/Person | Object, Event | \"What is the farmer wearing?\"",
        ]


# Auto-register on module import
node_registry.register(ObjectNodeType())
