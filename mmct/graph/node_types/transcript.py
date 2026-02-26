"""Transcript node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.graph.registry import node_registry


class TranscriptNodeType(BaseNodeType):
    """Transcript node - raw verbal content for verbal-only queries.
    
    Transcripts contain the raw speech/dialogue from video segments.
    Use for quote search, spoken content lookup, and queries where
    visual context is explicitly NOT needed.
    
    Note: Transcript and Chapter are 1:1 but have different purposes:
    - Transcript: verbal-only (speech)
    - Chapter: multimodal (visual + verbal summary)
    """
    
    @property
    def name(self) -> str:
        return "Transcript"
    
    @property
    def id_prefix(self) -> str:
        return "transcript_"
    
    @property
    def model_class(self) -> Type:
        from mmct.video_pipeline.core.ingestion.models import GraphTranscript
        return GraphTranscript
    
    @property
    def neo4j_properties(self) -> List[str]:
        return [
            "node_id", "video_id", "chunk_index", "start_time", "end_time",
            "video_duration", "transcript"
        ]
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        return attrs.get("transcript", "") or ""
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        return {
            "video_id": instance.video_id or "",
            "chunk_index": instance.chunk_index or 0,
            "transcript": instance.transcript or "",
            "start_time": instance.start_time or 0.0,
            "end_time": instance.end_time or 0.0,
            "video_duration": instance.video_duration or 0.0,
        }
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "transcript": (props.get("transcript", "") or "")[:500],
            "start_time": props.get("start_time"),
            "end_time": props.get("end_time"),
            "chunk_index": props.get("chunk_index"),
            "video_duration": props.get("video_duration"),
        }
    
    @property
    def time_property(self) -> str:
        return "start_time"
    
    @property
    def supports_time_range_filter(self) -> bool:
        return True
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Chapter": EdgeDefinition("HAS_TRANSCRIPT", "Chapter", "in"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        return TemporalEdgeDefinition(
            next_edge="NEXT_TRANSCRIPT",
            prev_edge="PREV_TRANSCRIPT",
            order_by="chunk_index",
        )
    
    @property
    def description(self) -> str:
        return "Raw verbal content (speech only) - use for quote search, spoken content"
    
    @property
    def use_cases(self) -> List[str]:
        return [
            "Quote/Spoken content | Transcript | \"What did the speaker say about X?\"",
            "Verbal-only query | Transcript | \"What was mentioned about fertilizer?\"",
        ]


# Auto-register on module import
node_registry.register(TranscriptNodeType())
