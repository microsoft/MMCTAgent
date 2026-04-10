"""Keyframe node type definition."""

from typing import Dict, Any, List, Type, Optional

from mmct.video_pipeline.core.graph.base import BaseNodeType, EdgeDefinition, TemporalEdgeDefinition
from mmct.video_pipeline.core.graph.registry import node_registry


class KeyframeNodeType(BaseNodeType):
    """Keyframe node - visual frames extracted from video chapters.
    
    Keyframes are representative images from video segments used for
    visual analysis. They have image embeddings (CLIP) for visual search.
    
    Note: Keyframes use image embeddings (CLIP) instead of text embeddings,
    so they are not searchable via the text vector search pipeline.
    """
    
    @property
    def name(self) -> str:
        return "Keyframe"
    
    @property
    def id_prefix(self) -> str:
        return "kf_"
    
    @property
    def model_class(self) -> Type:
        from mmct.video_pipeline.core.ingestion.models import GraphKeyframe
        return GraphKeyframe
    
    @property
    def neo4j_properties(self) -> List[str]:
        return [
            "node_id", "video_id", "chapter_id", "timestamp", "frame_index",
            "blob_url", "blob_name", "filepath", "motion_score"
        ]
    
    @property
    def embedding_dimension(self) -> int:
        return 512  # QdrantCLIP image embeddings
    
    @property
    def embedding_index_name(self) -> str:
        return "keyframe_embedding_index_image"
    
    @property
    def is_searchable(self) -> bool:
        # Keyframes use image embeddings, not text - exclude from text vector search
        return False
    
    def get_embedding_text(self, attrs: Dict[str, Any]) -> str:
        # Keyframes use image embeddings, not text
        # Return filepath for image embedding generation
        return attrs.get("filepath", "")
    
    def create_node_properties(self, instance) -> Dict[str, Any]:
        return {
            "video_id": instance.video_id or "",
            "chapter_id": instance.chapter_id or "",
            "timestamp": instance.timestamp or 0.0,
            "frame_index": instance.frame_index or 0,
            "filepath": instance.filepath or "",
            "blob_name": instance.blob_name or "",
            "blob_url": instance.blob_url or "",
            "motion_score": instance.motion_score or 0.0,
        }
    
    def format_search_result(self, props: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "blob_url": props.get("blob_url", ""),
            "timestamp": props.get("timestamp"),
            "frame_index": props.get("frame_index"),
            "chapter_id": props.get("chapter_id"),
            "motion_score": props.get("motion_score"),
        }
    
    @property
    def time_property(self) -> str:
        return "timestamp"
    
    @property
    def supports_time_range_filter(self) -> bool:
        return False  # Point-in-time
    
    def get_incoming_edges(self) -> Dict[str, EdgeDefinition]:
        return {
            "Chapter": EdgeDefinition("HAS_KEYFRAME", "Chapter", "in"),
        }
    
    def get_temporal_edges(self) -> Optional[TemporalEdgeDefinition]:
        return None  # Keyframes don't have NEXT/PREV edges
    
    @property
    def description(self) -> str:
        return "Visual frames linked to chapters"
    
    @property
    def use_cases(self) -> List[str]:
        return [
            "Visual detail | Keyframe + ImageAgent | \"What color is the tractor?\"",
        ]


# Auto-register on module import
node_registry.register(KeyframeNodeType())
