"""Graph Construction pipeline step.

Builds hierarchical temporal knowledge graphs from extracted events, objects,
chapters, and chapter groups for graph-based retrieval.

Node Types:
- ChapterGroup: High-level topic groupings
- Chapter: Video segments with temporal boundaries (multimodal summary)
- Transcript: Raw verbal content for targeted verbal-only retrieval
- Event: Atomic actions/occurrences within chapters
- Object: Entities that appear in events

Edge Types Created:
- NEXT_GROUP/PREV_GROUP: ChapterGroup ↔ ChapterGroup temporal sequence
- HAS_CHAPTER/IN_GROUP: ChapterGroup ↔ Chapter containment
- NEXT_CHAPTER/PREV_CHAPTER: Chapter ↔ Chapter temporal sequence
- HAS_TRANSCRIPT: Chapter → Transcript containment
- NEXT_TRANSCRIPT/PREV_TRANSCRIPT: Transcript ↔ Transcript temporal sequence
- HAS_EVENT/IN_CHAPTER: Chapter ↔ Event containment
- NEXT_EVENT/PREV_EVENT: Event ↔ Event temporal sequence
- CONTAINS/APPEARS_IN: Event ↔ Object participation
- SIMILAR_TO: Event ↔ Event semantic similarity
- CAUSES/RESULT_OF: Event → Event causal relationships

Note: Object deduplication is handled upstream in the temporal_graph step.
Events arrive with linked_object_ids already mapped to canonical objects.
"""

from typing import List, Dict, Any, Optional

from loguru import logger

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.video_pipeline.core.ingestion.models import (
    GraphEvent,
    GraphObject,
    GraphChapter,
    GraphChapterGroup,
    GraphKeyframe,
    GraphTranscript,
)
from mmct.providers.base import BaseLLMProvider
from mmct.providers.base.graph_db_provider import BaseGraphDBProvider
from mmct.providers.custom_providers.networkx_graph_provider import NetworkXGraphProvider

from .builder import GraphBuilder, GraphBuildResult
from .linker import GraphLinker, GraphLinkResult
from .causal_linker import CausalLinker, CausalLinkResult


# Default configuration values
DEFAULT_BATCH_SIZE = 50
DEFAULT_EVENT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MAX_SIMILAR_LINKS = 5

# Causal linking defaults
DEFAULT_MAX_CAUSAL_LINKS = 3
DEFAULT_CAUSAL_HIGH_THRESHOLD = 0.8
DEFAULT_CAUSAL_LOW_THRESHOLD = 0.5
DEFAULT_CAUSAL_MAX_GAP_SECONDS = 30.0
DEFAULT_CAUSAL_LLM_BATCH_SIZE = 8


@register_step("ingestion.graph_construction")
class GraphConstructionStep(PipelineStep):
    """Pipeline step for building hierarchical temporal knowledge graphs.
    
    Creates a complete graph hierarchy:
    ChapterGroup → Chapter → Event ↔ Object
                 ↘ Keyframe (visual frame linked to chapter)
                 ↘ Transcript (verbal content for targeted retrieval)
    
    Node Types:
    - ChapterGroup: High-level topic groupings (from chapter_grouping step)
    - Chapter: Video segments with multimodal summary (from chapters step)
    - Transcript: Raw verbal content for verbal-only queries (created from Chapter.transcript)
    - Event: Atomic actions (from temporal_graph step)
    - Object: Entities (from temporal_graph step)
    - Keyframe: Visual frames (from keyframes step)
    
    Edge Types (GraphBuilder):
    - NEXT_GROUP/PREV_GROUP: ChapterGroup temporal sequence
    - HAS_CHAPTER/IN_GROUP: ChapterGroup ↔ Chapter hierarchy
    - NEXT_CHAPTER/PREV_CHAPTER: Chapter temporal sequence
    - HAS_TRANSCRIPT: Chapter → Transcript hierarchy
    - NEXT_TRANSCRIPT/PREV_TRANSCRIPT: Transcript temporal sequence
    - HAS_EVENT/IN_CHAPTER: Chapter ↔ Event hierarchy
    - HAS_KEYFRAME/KEYFRAME_IN_CHAPTER: Chapter ↔ Keyframe hierarchy
    - NEXT_EVENT/PREV_EVENT: Event temporal sequence
    - CONTAINS/APPEARS_IN: Event ↔ Object bidirectional
    
    Additional Edges:
    - SIMILAR_TO: Event semantic similarity (GraphLinker)
    - CAUSES/RESULT_OF: Event causal relationships (CausalLinker)
    
    Params:
        source_step: Step ID for events/objects (default: "temporal_graph")
        source_chapters_step: Step ID for chapters (default: "chapters")
        source_groups_step: Step ID for chapter groups (default: "chapter_grouping")
        source_keyframes_step: Step ID for keyframes (default: "keyframes")
        graph_provider: Graph provider type (default: "networkx")
        batch_size: Batch size for node/edge creation (default: 50)
        event_similarity_threshold: Min similarity for SIMILAR_TO (default: 0.75)
        max_similar_links_per_event: Max SIMILAR_TO edges per event (default: 5)
        enable_similarity_linking: Whether to create SIMILAR_TO edges (default: True)
        persist_path: Path to persist graph (optional, for NetworkX)
        enable_causal_linking: Whether to create CAUSES/RESULT_OF edges (default: True)
        max_causal_links_per_event: Max causal edges per event (default: 3)
        causal_high_confidence_threshold: Score threshold for automatic acceptance (default: 0.8)
        causal_low_confidence_threshold: Score threshold for rejection (default: 0.5)
        causal_llm_validation: Whether to use LLM for uncertain candidates (default: True)
        causal_max_temporal_gap_seconds: Max time gap for causal relationships (default: 30.0)
    """
    
    step_type = "ingestion.graph_construction"
    description = "Build hierarchical temporal knowledge graph."
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute hierarchical graph construction.
        
        Args:
            context: Pipeline step context with data store and providers.
            
        Returns:
            StepResult containing graph statistics and metrics.
        """
        # Get source steps
        source_step: str = self.get_param(
            "source_step", context, default="temporal_graph"
        )
        source_chapters_step: str = self.get_param(
            "source_chapters_step", context, default="chapters"
        )
        source_groups_step: str = self.get_param(
            "source_groups_step", context, default="chapter_grouping"
        )
        source_keyframes_step: str = self.get_param(
            "source_keyframes_step", context, default="keyframes"
        )
        
        # Get events and objects from temporal_graph step
        events_data: List[Dict[str, Any]] = (
            context.data_store.get(source_step, "events") or []
        )
        objects_data: List[Dict[str, Any]] = (
            context.data_store.get(source_step, "objects") or []
        )
        
        # Also try to get model instances directly
        graph_events: Optional[List[GraphEvent]] = (
            context.data_store.get(source_step, "graph_events")
        )
        graph_objects: Optional[List[GraphObject]] = (
            context.data_store.get(source_step, "graph_objects")
        )
        
        # Get chapters - prefer from chapter_grouping (has group_index), fallback to dense_chapters
        chapters_data: List[Dict[str, Any]] = (
            context.data_store.get(source_groups_step, "chapters")  # Updated chapters with group_index
            or context.data_store.get(source_chapters_step, "raw_chapters")
            or context.data_store.get(source_chapters_step, "chapters")
            or context.data_store.get("dense_chapters", "raw_chapters")
            or context.data_store.get("dense_chapters", "chapters")
            or []
        )
        
        # Get chapter groups from chapter_grouping step
        groups_data: List[Dict[str, Any]] = (
            context.data_store.get(source_groups_step, "chapter_groups") or []
        )
        graph_groups: Optional[List[GraphChapterGroup]] = (
            context.data_store.get(source_groups_step, "groups")
        )
        
        # Get keyframes from dense_keyframes step
        keyframes_data: List[Dict[str, Any]] = (
            context.data_store.get(source_keyframes_step, "keyframes_per_chunk")
            or context.data_store.get("dense_keyframes", "keyframes_per_chunk")
            or []
        )
        
        # Get video_id early for model creation
        video_id: str = getattr(context, "video_id", "")
        video_duration: float = getattr(context, "video_duration", 0.0)
        
        # Convert dictionaries to model instances
        events = self._ensure_event_models(events_data, graph_events)
        objects = self._ensure_object_models(objects_data, graph_objects)
        chapters = self._ensure_chapter_models(chapters_data, video_id, video_duration)
        chapter_groups = self._ensure_chapter_group_models(groups_data, graph_groups, video_duration)
        keyframes = self._ensure_keyframe_models(keyframes_data, chapters, video_id)
        
        # Create transcript nodes from chapters (1:1 mapping)
        transcripts = self._create_transcript_models(chapters, video_id)
        
        if not events and not objects and not chapters:
            logger.warning("No events, objects, or chapters found for graph construction")
            return StepResult(
                step_id=self.step_id,
                outputs={
                    "graph_built": False,
                    "event_nodes": 0,
                    "object_nodes": 0,
                    "edges": {},
                },
                metrics={
                    "event_nodes_created": 0,
                    "object_nodes_created": 0,
                    "total_edges_created": 0,
                },
            )
        
        logger.info(
            f"Starting graph construction for video {video_id}: "
            f"{len(chapter_groups)} groups, {len(chapters)} chapters, "
            f"{len(transcripts)} transcripts, {len(events)} events, "
            f"{len(objects)} objects, {len(keyframes)} keyframes"
        )
        
        # Get or create graph provider
        graph_provider = self._get_graph_provider(context)
        
        # Clear any existing graph data to avoid duplicates from previous runs
        await graph_provider.clear_database()
        
        # Get configuration parameters
        batch_size: int = self.get_param(
            "batch_size", context, default=DEFAULT_BATCH_SIZE
        )
        event_sim_threshold: float = self.get_param(
            "event_similarity_threshold", context, default=DEFAULT_EVENT_SIMILARITY_THRESHOLD
        )
        max_similar_links: int = self.get_param(
            "max_similar_links_per_event", context, default=DEFAULT_MAX_SIMILAR_LINKS
        )
        enable_similarity: bool = self.get_param(
            "enable_similarity_linking", context, default=True
        )
        
        # Build hierarchical graph nodes and edges
        builder = GraphBuilder(
            graph_provider=graph_provider,
            batch_size=batch_size,
        )
        
        build_result: GraphBuildResult = await builder.build_graph(
            events=events,
            objects=objects,
            video_id=video_id,
            chapters=chapters,
            chapter_groups=chapter_groups,
            keyframes=keyframes,
            transcripts=transcripts,
        )
        
        # Create semantic similarity links
        link_result = GraphLinkResult()
        
        if enable_similarity and events:
            linker = GraphLinker(
                graph_provider=graph_provider,
                event_similarity_threshold=event_sim_threshold,
                max_similar_links_per_event=max_similar_links,
                batch_size=batch_size,
            )
            
            link_result = await linker.link_graph(
                events=events,
                video_id=video_id,
            )
        
        # Causal linking configuration
        enable_causal: bool = self.get_param(
            "enable_causal_linking", context, default=True
        )
        max_causal_links: int = self.get_param(
            "max_causal_links_per_event", context, default=DEFAULT_MAX_CAUSAL_LINKS
        )
        causal_high_threshold: float = self.get_param(
            "causal_high_confidence_threshold", context, default=DEFAULT_CAUSAL_HIGH_THRESHOLD
        )
        causal_low_threshold: float = self.get_param(
            "causal_low_confidence_threshold", context, default=0.65  # Raised default
        )
        causal_llm_validation: bool = self.get_param(
            "causal_llm_validation", context, default=True
        )
        causal_max_gap: float = self.get_param(
            "causal_max_temporal_gap_seconds", context, default=DEFAULT_CAUSAL_MAX_GAP_SECONDS
        )
        skip_adjacent_events: bool = self.get_param(
            "skip_adjacent_events", context, default=True
        )
        apply_transitive_reduction: bool = self.get_param(
            "apply_transitive_reduction", context, default=True
        )
        
        # Create causal links
        causal_result = CausalLinkResult()
        
        if enable_causal and events:
            llm_provider = self._get_llm_provider(context) if causal_llm_validation else None
            
            causal_linker = CausalLinker(
                graph_provider=graph_provider,
                llm_provider=llm_provider,
                max_causal_links_per_event=max_causal_links,
                heuristic_high_threshold=causal_high_threshold,
                heuristic_low_threshold=causal_low_threshold,
                batch_size=batch_size,
                llm_batch_size=DEFAULT_CAUSAL_LLM_BATCH_SIZE,
                max_temporal_gap_seconds=causal_max_gap,
                skip_adjacent_events=skip_adjacent_events,
                apply_transitive_reduction=apply_transitive_reduction,
            )
            
            causal_result = await causal_linker.link_causal_relationships(
                events=events,
                video_id=video_id,
            )
        
        # Aggregate results
        total_edges = (
            build_result.group_temporal_edges_created
            + build_result.chapter_temporal_edges_created
            + build_result.transcript_temporal_edges_created
            + build_result.event_temporal_edges_created
            + build_result.hierarchy_edges_created
            + build_result.contains_edges_created
            + build_result.keyframe_edges_created
            + build_result.has_transcript_edges_created
            + link_result.similar_edges_created
            + causal_result.causes_edges_created
            + causal_result.result_of_edges_created
        )
        
        all_errors = build_result.errors + link_result.errors + causal_result.errors
        if all_errors:
            for error in all_errors[:5]:
                logger.warning(f"Graph construction error: {error}")
        
        logger.info(
            f"Graph construction complete for video {video_id}: "
            f"{build_result.chapter_group_nodes_created} groups, "
            f"{build_result.chapter_nodes_created} chapters, "
            f"{build_result.transcript_nodes_created} transcripts, "
            f"{build_result.keyframe_nodes_created} keyframes, "
            f"{build_result.event_nodes_created} events, "
            f"{build_result.object_nodes_created} objects, "
            f"{total_edges} edges"
        )
        
        return StepResult(
            step_id=self.step_id,
            outputs={
                "graph_built": True,
                "chapter_group_nodes": build_result.chapter_group_nodes_created,
                "chapter_nodes": build_result.chapter_nodes_created,
                "transcript_nodes": build_result.transcript_nodes_created,
                "keyframe_nodes": build_result.keyframe_nodes_created,
                "event_nodes": build_result.event_nodes_created,
                "object_nodes": build_result.object_nodes_created,
                "similarity_linking_enabled": enable_similarity,
                "causal_linking_enabled": enable_causal,
                "edges": {
                    "NEXT_GROUP_PREV_GROUP": build_result.group_temporal_edges_created,
                    "NEXT_CHAPTER_PREV_CHAPTER": build_result.chapter_temporal_edges_created,
                    "NEXT_TRANSCRIPT_PREV_TRANSCRIPT": build_result.transcript_temporal_edges_created,
                    "HAS_CHAPTER_HAS_EVENT": build_result.hierarchy_edges_created,
                    "NEXT_EVENT_PREV_EVENT": build_result.event_temporal_edges_created,
                    "CONTAINS": build_result.contains_edges_created,
                    "HAS_KEYFRAME": build_result.keyframe_edges_created,
                    "HAS_TRANSCRIPT": build_result.has_transcript_edges_created,
                    "SIMILAR_TO": link_result.similar_edges_created,
                    "CAUSES": causal_result.causes_edges_created,
                    "RESULT_OF": causal_result.result_of_edges_created,
                },
                "causal_result": {
                    "causes_edges": causal_result.causes_edges_created,
                    "result_of_edges": causal_result.result_of_edges_created,
                    "pairs_validated_by_llm": causal_result.pairs_validated_by_llm,
                    "pairs_rejected": causal_result.pairs_rejected,
                    "high_confidence_pairs": causal_result.high_confidence_pairs,
                },
                "graph_provider": graph_provider,
                "errors": all_errors,
            },
            metrics={
                "chapter_group_nodes_created": build_result.chapter_group_nodes_created,
                "chapter_nodes_created": build_result.chapter_nodes_created,
                "transcript_nodes_created": build_result.transcript_nodes_created,
                "keyframe_nodes_created": build_result.keyframe_nodes_created,
                "event_nodes_created": build_result.event_nodes_created,
                "object_nodes_created": build_result.object_nodes_created,
                "causal_linking_enabled": 1 if enable_causal else 0,
                "group_temporal_edges_created": build_result.group_temporal_edges_created,
                "chapter_temporal_edges_created": build_result.chapter_temporal_edges_created,
                "transcript_temporal_edges_created": build_result.transcript_temporal_edges_created,
                "has_transcript_edges_created": build_result.has_transcript_edges_created,
                "hierarchy_edges_created": build_result.hierarchy_edges_created,
                "keyframe_edges_created": build_result.keyframe_edges_created,
                "event_temporal_edges_created": build_result.event_temporal_edges_created,
                "contains_edges_created": build_result.contains_edges_created,
                "similar_edges_created": link_result.similar_edges_created,
                "causes_edges_created": causal_result.causes_edges_created,
                "result_of_edges_created": causal_result.result_of_edges_created,
                "causal_pairs_validated_by_llm": causal_result.pairs_validated_by_llm,
                "causal_pairs_rejected": causal_result.pairs_rejected,
                "causal_high_confidence_pairs": causal_result.high_confidence_pairs,
                "total_edges_created": total_edges,
                "error_count": len(all_errors),
            },
        )
    
    def _get_graph_provider(self, context: StepContext) -> BaseGraphDBProvider:
        """Get or create graph database provider.
        
        Args:
            context: Pipeline step context.
            
        Returns:
            Graph database provider instance.
        """
        # Check if provider is configured on context
        if hasattr(context.provider, "graph_provider"):
            configured = getattr(context.provider, "graph_provider", None)
            if configured is not None:
                return configured
        
        # Get configuration from params
        provider_type: str = self.get_param(
            "graph_provider", context, default="networkx"
        )
        persist_path: Optional[str] = self.get_param(
            "persist_path", context, default=None
        )
        
        video_id: str = getattr(context, "video_id", "default")
        
        if provider_type == "networkx":
            return NetworkXGraphProvider(
                database_name=video_id,
                persist_path=persist_path,
            )
        
        # Default to NetworkX for local development
        logger.info(f"Using NetworkX graph provider for video {video_id}")
        return NetworkXGraphProvider(
            database_name=video_id,
            persist_path=persist_path,
        )
    
    def _get_llm_provider(self, context: StepContext) -> Optional[BaseLLMProvider]:
        """Get LLM provider from context if available.
        
        Args:
            context: Pipeline step context.
            
        Returns:
            LLM provider instance if available, None otherwise.
        """
        # Check if provider is configured on context.provider
        if hasattr(context.provider, "llm_provider"):
            provider = getattr(context.provider, "llm_provider", None)
            if provider is not None:
                return provider
        
        # Check alternative attribute name
        if hasattr(context.provider, "llm"):
            provider = getattr(context.provider, "llm", None)
            if provider is not None:
                return provider
        
        logger.debug("No LLM provider available for causal validation")
        return None
    
    def _ensure_event_models(
        self,
        events_data: List[Dict[str, Any]],
        graph_events: Optional[List[GraphEvent]],
    ) -> List[GraphEvent]:
        """Ensure events are GraphEvent model instances.
        
        Args:
            events_data: Event dictionaries from data store.
            graph_events: Optional GraphEvent instances from data store.
            
        Returns:
            List of GraphEvent instances.
        """
        # Prefer model instances if available
        if graph_events:
            return graph_events
        
        # Convert dictionaries to models
        events = []
        for data in events_data:
            try:
                event = GraphEvent(**data)
                events.append(event)
            except Exception as e:
                logger.warning(f"Failed to convert event data: {e}")
        
        return events
    
    def _ensure_object_models(
        self,
        objects_data: List[Dict[str, Any]],
        graph_objects: Optional[List[GraphObject]],
    ) -> List[GraphObject]:
        """Ensure objects are GraphObject model instances.
        
        Args:
            objects_data: Object dictionaries from data store.
            graph_objects: Optional GraphObject instances from data store.
            
        Returns:
            List of GraphObject instances.
        """
        # Prefer model instances if available
        if graph_objects:
            return graph_objects
        
        # Convert dictionaries to models
        objects = []
        for data in objects_data:
            try:
                obj = GraphObject(**data)
                objects.append(obj)
            except Exception as e:
                logger.warning(f"Failed to convert object data: {e}")
        
        return objects
    
    def _ensure_chapter_models(
        self,
        chapters_data: List[Dict[str, Any]],
        video_id: str = "",
        video_duration: float = 0.0,
    ) -> List[GraphChapter]:
        """Convert chapter dictionaries to GraphChapter model instances.
        
        Args:
            chapters_data: Chapter dictionaries from data store.
            video_id: Video identifier from context.
            video_duration: Total duration of the video in seconds.
            
        Returns:
            List of GraphChapter instances.
        """
        chapters = []
        
        for idx, data in enumerate(chapters_data):
            try:
                # Use video_id from context, fallback to data
                vid = video_id or data.get("video_id", "")
                
                # Handle chunk_index - fallback to idx if None or missing
                chunk_idx = data.get("chunk_index")
                if chunk_idx is None:
                    chunk_idx = idx
                
                # Get video_duration from data first (might be set by upstream step), fallback to context
                duration = data.get("video_duration") or video_duration
                
                chapter = GraphChapter(
                    id=f"chapter_{vid}_{idx:03d}" if vid else f"chapter_{idx:03d}",
                    video_id=vid,
                    chunk_index=chunk_idx,
                    start_time=data.get("start"),
                    end_time=data.get("end"),
                    video_duration=duration,
                    timestamped_description=data.get("timestamped_description"),
                    summary=data.get("summary") or data.get("raw_summary", {}).get("summary"),
                    scene_composition=self._format_scene_composition(data.get("scene_composition")),
                    ocr_data=self._format_ocr_data(data.get("ocr_data")),
                    transcript=data.get("transcript"),
                    group_index=data.get("group_index"),
                    metadata=data.get("metadata"),
                )
                chapters.append(chapter)
                logger.debug(f"Created chapter {chapter.id} with chunk_index={chunk_idx}")
            except Exception as e:
                logger.warning(f"Failed to convert chapter data at index {idx}: {e}")
        
        logger.info(f"Created {len(chapters)} chapters with chunk_indexes: {[c.chunk_index for c in chapters]}")
        return chapters
    
    @staticmethod
    def _format_scene_composition(scene) -> Optional[str]:
        """Convert SceneComposition dict to a human-readable string."""
        if not scene:
            return None
        if isinstance(scene, str):
            return scene
        if hasattr(scene, "model_dump"):
            scene = scene.model_dump()
        parts = []
        if scene.get("environment_type"):
            parts.append(scene["environment_type"])
        if scene.get("lighting_conditions"):
            parts.append(scene["lighting_conditions"])
        if scene.get("camera_angle"):
            parts.append(scene["camera_angle"])
        regions = scene.get("spatial_regions")
        if regions and isinstance(regions, dict):
            region_parts = []
            for region, items in regions.items():
                if items:
                    region_parts.append(f"{region}: {', '.join(items)}")
            if region_parts:
                parts.append("layout: " + "; ".join(region_parts))
        return ", ".join(parts) if parts else None

    @staticmethod
    def _format_ocr_data(ocr) -> Optional[str]:
        """Convert ChapterOCRData dict to a human-readable string."""
        if not ocr:
            return None
        if isinstance(ocr, str):
            return ocr
        if hasattr(ocr, "model_dump"):
            ocr = ocr.model_dump()
        parts = []
        for ext in ocr.get("extractions", []):
            text = ext.get("text", "")
            text_type = ext.get("text_type", "")
            if text:
                parts.append(f"{text_type}: {text}" if text_type else text)
        persistent = ocr.get("persistent_text", [])
        if persistent:
            parts.append(f"persistent: {', '.join(persistent)}")
        return "; ".join(parts) if parts else None
    
    def _ensure_chapter_group_models(
        self,
        groups_data: List[Dict[str, Any]],
        graph_groups: Optional[List[GraphChapterGroup]],
        video_duration: float = 0.0,
    ) -> List[GraphChapterGroup]:
        """Ensure chapter groups are GraphChapterGroup model instances.
        
        Args:
            groups_data: Group dictionaries from data store.
            graph_groups: Optional GraphChapterGroup instances from data store.
            video_duration: Total duration of the video in seconds.
            
        Returns:
            List of GraphChapterGroup instances.
        """
        # Prefer model instances if available
        if graph_groups:
            # Convert ChapterGroup to GraphChapterGroup if needed
            result = []
            for group in graph_groups:
                if isinstance(group, GraphChapterGroup):
                    # Ensure video_duration is set
                    if not group.video_duration:
                        group.video_duration = video_duration
                    result.append(group)
                else:
                    # Convert from ChapterGroup model
                    try:
                        graph_group = GraphChapterGroup(
                            id=getattr(group, "id", None),
                            video_id=getattr(group, "video_id", None),
                            name=getattr(group, "name", None),
                            order=getattr(group, "order", None),
                            chapter_indices=getattr(group, "chapter_indices", None),
                            start_time=getattr(group, "start_time", None),
                            end_time=getattr(group, "end_time", None),
                            video_duration=getattr(group, "video_duration", None) or video_duration,
                            summary=getattr(group, "summary", None),
                            topics=getattr(group, "topics", None),
                            metadata=getattr(group, "metadata", None),
                        )
                        result.append(graph_group)
                    except Exception as e:
                        logger.warning(f"Failed to convert group model: {e}")
            return result
        
        # Convert dictionaries to models
        groups = []
        video_id = ""
        
        for idx, data in enumerate(groups_data):
            try:
                if not video_id:
                    video_id = data.get("video_id", "")
                
                group = GraphChapterGroup(
                    id=data.get("id") or f"group_{video_id}_{idx:03d}" if video_id else f"group_{idx:03d}",
                    video_id=video_id or data.get("video_id"),
                    name=data.get("name"),
                    order=data.get("order", idx),
                    chapter_indices=data.get("chapter_indices"),
                    start_time=data.get("start_time"),
                    end_time=data.get("end_time"),
                    video_duration=data.get("video_duration") or video_duration,
                    summary=data.get("summary"),
                    topics=data.get("topics"),
                    metadata=data.get("metadata"),
                )
                groups.append(group)
            except Exception as e:
                logger.warning(f"Failed to convert group data at index {idx}: {e}")
        
        return groups
    
    def _ensure_keyframe_models(
        self,
        keyframes_data: List[Dict[str, Any]],
        chapters: List[GraphChapter],
        video_id: str = "",
    ) -> List[GraphKeyframe]:
        """Convert keyframe data to GraphKeyframe model instances.
        
        Keyframes are organized by chunk. Each chunk's keyframes are linked
        to the corresponding chapter by chunk_index.
        
        Args:
            keyframes_data: List of chunk data, each with keyframes list.
            chapters: List of GraphChapter instances for chapter_id mapping.
            video_id: Video identifier from context.
            
        Returns:
            List of GraphKeyframe instances.
        """
        keyframes = []
        
        # Build chapter lookup by chunk_index
        chapter_by_chunk: Dict[int, GraphChapter] = {}
        for chapter in chapters:
            if chapter.chunk_index is not None:
                chapter_by_chunk[chapter.chunk_index] = chapter
        
        logger.debug(f"Chapter lookup by chunk_index: {list(chapter_by_chunk.keys())}")
        
        for chunk_data in keyframes_data:
            chunk_id = chunk_data.get("chunk_id")
            chunk_keyframes = chunk_data.get("keyframes", [])
            
            # Parse chunk_id to integer for chapter lookup
            # chunk_id can be int (0, 1, 2) or string ("0", "1", "2")
            if chunk_id is not None:
                try:
                    chunk_index = int(chunk_id)
                except (ValueError, TypeError):
                    chunk_index = -1
            else:
                chunk_index = -1
            
            chapter = chapter_by_chunk.get(chunk_index)
            chapter_id = chapter.id if chapter else None
            # Use video_id from context, fallback to chapter
            vid = video_id or (chapter.video_id if chapter else "")
            
            if not chapter_id:
                logger.warning(
                    f"No chapter found for chunk_id={chunk_id} (chunk_index={chunk_index}). "
                    f"Available: {list(chapter_by_chunk.keys())}"
                )
            
            for kf_idx, kf_data in enumerate(chunk_keyframes):
                try:
                    keyframe = GraphKeyframe(
                        id=f"kf_{vid}_{chunk_index}_{kf_idx:03d}" if vid else f"kf_{chunk_index}_{kf_idx:03d}",
                        video_id=vid,
                        chapter_id=chapter_id,
                        timestamp=kf_data.get("timestamp"),
                        frame_index=kf_data.get("frame_index", kf_idx),
                        filepath=kf_data.get("filepath"),
                        blob_name=kf_data.get("blob_name"),
                        blob_url=kf_data.get("blob_url"),
                        motion_score=kf_data.get("motion_score"),
                        metadata=kf_data.get("metadata"),
                    )
                    keyframes.append(keyframe)
                except Exception as e:
                    logger.warning(f"Failed to convert keyframe at chunk {chunk_index}, index {kf_idx}: {e}")
        
        linked_count = sum(1 for kf in keyframes if kf.chapter_id)
        logger.info(f"Created {len(keyframes)} keyframes, {linked_count} linked to chapters")
        return keyframes
    
    def _create_transcript_models(
        self,
        chapters: List[GraphChapter],
        video_id: str = "",
    ) -> List[GraphTranscript]:
        """Create GraphTranscript instances from chapters.
        
        Creates 1:1 mapping between chapters and transcripts. Each transcript
        contains the raw verbal content for targeted verbal-only retrieval.
        
        For chapters without transcript text, creates a placeholder node with
        "No transcript available for this video segment" to enable fallback.
        
        Args:
            chapters: List of GraphChapter instances with transcript data.
            video_id: Video identifier from context.
            
        Returns:
            List of GraphTranscript instances.
        """
        transcripts = []
        
        for chapter in chapters:
            try:
                chunk_idx = chapter.chunk_index if chapter.chunk_index is not None else 0
                vid = video_id or chapter.video_id or ""
                
                # Get transcript text or use placeholder for no-transcript videos
                transcript_text = chapter.transcript
                if not transcript_text or transcript_text.strip() == "":
                    transcript_text = "No transcript available for this video segment"
                
                transcript = GraphTranscript(
                    id=f"transcript_{vid}_{chunk_idx:03d}" if vid else f"transcript_{chunk_idx:03d}",
                    video_id=vid,
                    chunk_index=chunk_idx,
                    transcript=transcript_text,
                    start_time=chapter.start_time,
                    end_time=chapter.end_time,
                    video_duration=getattr(chapter, "video_duration", None),
                    metadata={"source_chapter_id": chapter.id},
                )
                transcripts.append(transcript)
            except Exception as e:
                logger.warning(f"Failed to create transcript for chapter {chapter.id}: {e}")
        
        logger.info(f"Created {len(transcripts)} transcript nodes from {len(chapters)} chapters")
        return transcripts
