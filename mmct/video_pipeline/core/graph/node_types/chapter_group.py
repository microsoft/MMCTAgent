"""ChapterGroup node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.video_pipeline.core.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.video_pipeline.core.graph.registry import node_registry


class ChapterGroupNodeType(BaseNodeType):
    """ChapterGroup node - high-level topic groupings of video sections.
    
    ChapterGroups represent semantic clusters of chapters that share
    a common theme or topic. Used for video overview and discovery.
    """
    
    @property
    def name(self) -> str:
        return "ChapterGroup"
    
    @property
    def id_prefix(self) -> str:
        return "chaptergroup_"
    
    @property
    def model_class(self) -> Type:
        from mmct.video_pipeline.core.ingestion.models import GraphChapterGroup
        return GraphChapterGroup
    
    @property
    def neo4j_properties(self) -> List[str]:
        return [
            "node_id", "video_id", "video_title", "name", "order", "start_time", "end_time",
            "video_duration", "summary", "topics", "chapter_indices"
        ]
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        return attrs.get("summary", "") or ""
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        return {
            "name": instance.name or "",
            "video_id": instance.video_id or "",
            "order": instance.order or 0,
            "start_time": instance.start_time or 0.0,
            "end_time": instance.end_time or 0.0,
            "video_duration": getattr(instance, "video_duration", None),
            "summary": instance.summary or "",
            "topics": instance.topics or [],
            "chapter_indices": instance.chapter_indices or [],
        }
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "video_title": props.get("video_title"),
            "summary": props.get("summary", "") or "",
            "start_time": props.get("start_time"),
            "end_time": props.get("end_time"),
            "video_duration": props.get("video_duration"),
            "order": props.get("order"),
        }
    
    @property
    def time_property(self) -> str:
        return "start_time"
    
    @property
    def supports_time_range_filter(self) -> bool:
        return True
    
    def get_outgoing_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Chapter": EdgeDefinition("HAS_CHAPTER", "Chapter", "out"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        return TemporalEdgeDefinition(
            next_edge="NEXT_GROUP",
            prev_edge="PREV_GROUP",
            order_by="order",
        )
    
    @property
    def description(self) -> str:
        return "High-level topic groupings (video sections)"
    
    @property
    def use_cases(self) -> List[str]:
        return [
            "Overview/Summary | ChapterGroup, Chapter | \"What is this video about?\"",
            "Cross-video | ChapterGroup (multi-video) | \"Which videos show farming?\"",
        ]


# Auto-register on module import
node_registry.register(ChapterGroupNodeType())
