"""Chapter node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.graph.registry import node_registry


class ChapterNodeType(BaseNodeType):
    """Chapter node - video segments with multimodal summaries.
    
    Chapters represent temporal segments of a video with both visual
    and verbal context summarized together. Use when visual context
    is needed alongside spoken content.
    """
    
    @property
    def name(self) -> str:
        return "Chapter"
    
    @property
    def id_prefix(self) -> str:
        return "chapter_"
    
    @property
    def model_class(self) -> Type:
        from mmct.video_pipeline.core.ingestion.models import GraphChapter
        return GraphChapter
    
    @property
    def neo4j_properties(self) -> List[str]:
        return [
            "node_id", "video_id", "chunk_index", "start_time", "end_time",
            "video_duration", "summary", "group_index"
        ]
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        return attrs.get("summary", "") or ""
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        return {
            "video_id": instance.video_id or "",
            "chunk_index": instance.chunk_index or 0,
            "start_time": instance.start_time or 0.0,
            "end_time": instance.end_time or 0.0,
            "video_duration": getattr(instance, "video_duration", None),
            "summary": instance.summary or "",
            "group_index": instance.group_index,
        }
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "summary": (props.get("summary", "") or "")[:500],
            "title": props.get("title", ""),
            "start_time": props.get("start_time"),
            "end_time": props.get("end_time"),
            "video_duration": props.get("video_duration"),
            "chunk_index": props.get("chunk_index"),
        }
    
    @property
    def time_property(self) -> str:
        return "start_time"
    
    @property
    def supports_time_range_filter(self) -> bool:
        return True
    
    def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Event": EdgeDefinition("HAS_EVENT", "Event", "out"),
            "Keyframe": EdgeDefinition("HAS_KEYFRAME", "Keyframe", "out"),
            "Transcript": EdgeDefinition("HAS_TRANSCRIPT", "Transcript", "out"),
        }
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "ChapterGroup": EdgeDefinition("HAS_CHAPTER", "ChapterGroup", "in"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        return TemporalEdgeDefinition(
            next_edge="NEXT_CHAPTER",
            prev_edge="PREV_CHAPTER",
            order_by="chunk_index",
        )
    
    @property
    def description(self) -> str:
        return "Video segments with multimodal summaries (visual + verbal cues)"
    
    @property
    def use_cases(self) -> List[str]:
        return [
            "Overview/Summary | ChapterGroup, Chapter | \"What is this video about?\"",
            "Specific action | Event, Chapter | \"How does he dig the soil?\"",
            "Timeline/Sequence | Event, Chapter | \"What happens after mixing?\"",
            "Temporal (time-based) | Chapter, Event + TIME FILTER | \"What happens in the first 2 minutes?\"",
        ]


# Auto-register on module import
node_registry.register(ChapterNodeType())
