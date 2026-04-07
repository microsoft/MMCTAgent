"""Event node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.graph.registry import node_registry


class EventNodeType(BaseNodeType):
    """Event node - atomic actions/occurrences with precise timestamps.
    
    Events represent specific actions or occurrences within a video
    with precise temporal boundaries. Used for queries about specific
    moments, sequences, and what happens at particular times.
    """
    
    @property
    def name(self) -> str:
        return "Event"
    
    @property
    def id_prefix(self) -> str:
        return "event_"
    
    @property
    def model_class(self) -> Type:
        from mmct.video_pipeline.core.ingestion.models import GraphEvent
        return GraphEvent
    
    @property
    def neo4j_properties(self) -> List[str]:
        return [
            "node_id", "video_id", "description", "timestamp", "duration",
            "event_type", "participants", "chapter_index", "sequence_number"
        ]
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        return attrs.get("description", "") or ""
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        return {
            "description": instance.description or "",
            "video_id": instance.video_id or "",
            "timestamp": instance.timestamp or 0.0,
            "duration": instance.duration or 0.0,
            "event_type": instance.event_type or "action",
            "participants": instance.participants or [],
            "chapter_index": instance.chapter_index,
            "sequence_number": instance.sequence_number,
        }
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "description": props.get("description", ""),
            "start_time": props.get("timestamp") or props.get("start_time"),
            "end_time": props.get("end_time"),
            "duration": props.get("duration"),
            "chapter_index": props.get("chapter_index"),
            "event_type": props.get("event_type"),
        }
    
    @property
    def time_property(self) -> str:
        return "timestamp"
    
    @property
    def supports_time_range_filter(self) -> bool:
        return False  # Events use point-in-time filtering
    
    def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Object": EdgeDefinition("CONTAINS", "Object", "out"),
        }
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Chapter": EdgeDefinition("HAS_EVENT", "Chapter", "in"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        return TemporalEdgeDefinition(
            next_edge="NEXT_EVENT",
            prev_edge="PREV_EVENT",
            order_by="sequence_number",
        )
    
    @property
    def description(self) -> str:
        return "Atomic actions/occurrences with timestamps"
    
    @property
    def use_cases(self) -> List[str]:
        return [
            "Specific action | Event, Chapter | \"How does he dig the soil?\"",
            "Timeline/Sequence | Event, Chapter | \"What happens after mixing?\"",
            "Temporal (time-based) | Chapter, Event + TIME FILTER | \"What happens in the first 2 minutes?\"",
        ]


# Auto-register on module import
node_registry.register(EventNodeType())
