"""Graph builder for constructing hierarchical temporal knowledge graphs.

Creates a complete hierarchical graph structure:

Node Types:
- ChapterGroup: High-level topic groupings
- Chapter: Video segments with temporal boundaries
- Transcript: Raw verbal content per chapter
- Event: Atomic actions/occurrences within chapters
- Object: Entities that appear in events
- Keyframe: Visual frames from chapters

Edge Types (Hierarchy - single direction, query both ways):
- HAS_CHAPTER: ChapterGroup → Chapter
- HAS_EVENT: Chapter → Event
- HAS_KEYFRAME: Chapter → Keyframe
- HAS_TRANSCRIPT: Chapter → Transcript
- CONTAINS: Event → Object

Edge Types (Temporal - bidirectional for navigation):
- NEXT_GROUP/PREV_GROUP: ChapterGroup ↔ ChapterGroup
- NEXT_CHAPTER/PREV_CHAPTER: Chapter ↔ Chapter
- NEXT_TRANSCRIPT/PREV_TRANSCRIPT: Transcript ↔ Transcript
- NEXT_EVENT/PREV_EVENT: Event ↔ Event

Note: Object deduplication is handled upstream in the temporal_graph step.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from loguru import logger

from mmct.video_pipeline.core.ingestion.models import (
    GraphEvent,
    GraphObject,
    GraphChapter,
    GraphChapterGroup,
    GraphKeyframe,
    GraphTranscript,
)
from mmct.providers.base.graph_db_provider import BaseGraphDBProvider


@dataclass
class GraphBuildResult:
    """Result of graph building operation."""
    
    # Node counts
    chapter_group_nodes_created: int = 0
    chapter_nodes_created: int = 0
    transcript_nodes_created: int = 0
    keyframe_nodes_created: int = 0
    event_nodes_created: int = 0
    object_nodes_created: int = 0
    
    # Edge counts
    group_temporal_edges_created: int = 0  # NEXT_GROUP/PREV_GROUP
    chapter_temporal_edges_created: int = 0  # NEXT_CHAPTER/PREV_CHAPTER
    transcript_temporal_edges_created: int = 0  # NEXT_TRANSCRIPT/PREV_TRANSCRIPT
    event_temporal_edges_created: int = 0  # NEXT_EVENT/PREV_EVENT
    hierarchy_edges_created: int = 0  # HAS_CHAPTER, HAS_EVENT
    keyframe_edges_created: int = 0  # HAS_KEYFRAME
    has_transcript_edges_created: int = 0  # HAS_TRANSCRIPT
    contains_edges_created: int = 0  # CONTAINS
    
    errors: List[str] = field(default_factory=list)


class GraphBuilder:
    """Builds hierarchical temporal knowledge graph nodes and edges.
    
    Creates a complete graph hierarchy with single-direction containment edges
    (query code handles bidirectional traversal):
    - ChapterGroup nodes with NEXT_GROUP/PREV_GROUP edges
    - Chapter nodes with NEXT_CHAPTER/PREV_CHAPTER edges
    - Transcript nodes with NEXT_TRANSCRIPT/PREV_TRANSCRIPT edges and HAS_TRANSCRIPT from Chapter
    - Keyframe nodes with HAS_KEYFRAME from Chapter
    - Event nodes with NEXT_EVENT/PREV_EVENT edges and HAS_EVENT from Chapter
    - Object nodes with CONTAINS from Event
    """
    
    # Node types
    NODE_TYPE_CHAPTER_GROUP = "ChapterGroup"
    NODE_TYPE_CHAPTER = "Chapter"
    NODE_TYPE_TRANSCRIPT = "Transcript"
    NODE_TYPE_KEYFRAME = "Keyframe"
    NODE_TYPE_EVENT = "Event"
    NODE_TYPE_OBJECT = "Object"
    
    # Event-Object edge (single direction: Event → Object)
    EDGE_CONTAINS = "CONTAINS"
    
    # Event temporal edges
    EDGE_NEXT_EVENT = "NEXT_EVENT"
    EDGE_PREV_EVENT = "PREV_EVENT"
    
    # Chapter temporal edges
    EDGE_NEXT_CHAPTER = "NEXT_CHAPTER"
    EDGE_PREV_CHAPTER = "PREV_CHAPTER"
    
    # Transcript temporal edges
    EDGE_NEXT_TRANSCRIPT = "NEXT_TRANSCRIPT"
    EDGE_PREV_TRANSCRIPT = "PREV_TRANSCRIPT"
    
    # Chapter group temporal edges
    EDGE_NEXT_GROUP = "NEXT_GROUP"
    EDGE_PREV_GROUP = "PREV_GROUP"
    
    # Hierarchy edges (single direction: parent → child)
    EDGE_HAS_CHAPTER = "HAS_CHAPTER"      # ChapterGroup → Chapter
    EDGE_HAS_EVENT = "HAS_EVENT"          # Chapter → Event
    EDGE_HAS_KEYFRAME = "HAS_KEYFRAME"    # Chapter → Keyframe
    EDGE_HAS_TRANSCRIPT = "HAS_TRANSCRIPT"  # Chapter → Transcript
    
    def __init__(
        self,
        graph_provider: BaseGraphDBProvider,
        batch_size: int = 50,
    ):
        """Initialize GraphBuilder.
        
        Args:
            graph_provider: Provider for graph database operations.
            batch_size: Number of items to process in each batch.
        """
        self.graph_provider = graph_provider
        self.batch_size = batch_size
    
    async def build_graph(
        self,
        events: List[GraphEvent],
        objects: List[GraphObject],
        video_id: str,
        chapters: Optional[List[GraphChapter]] = None,
        chapter_groups: Optional[List[GraphChapterGroup]] = None,
        keyframes: Optional[List[GraphKeyframe]] = None,
        transcripts: Optional[List[GraphTranscript]] = None,
    ) -> GraphBuildResult:
        """Build complete hierarchical graph.
        
        Creates all nodes and relationships in the hierarchy:
        ChapterGroup → Chapter → Event ↔ Object
                     ↘ Keyframe
                     ↘ Transcript
        
        Args:
            events: List of GraphEvent instances.
            objects: List of GraphObject instances (already deduplicated).
            video_id: Video identifier for scoping.
            chapters: Optional list of GraphChapter instances.
            chapter_groups: Optional list of GraphChapterGroup instances.
            keyframes: Optional list of GraphKeyframe instances.
            transcripts: Optional list of GraphTranscript instances.
            
        Returns:
            GraphBuildResult with counts and any errors.
        """
        result = GraphBuildResult()
        chapters = chapters or []
        chapter_groups = chapter_groups or []
        keyframes = keyframes or []
        transcripts = transcripts or []
        
        logger.info(
            f"Building hierarchical graph for video {video_id}: "
            f"{len(chapter_groups)} groups, {len(chapters)} chapters, "
            f"{len(transcripts)} transcripts, {len(keyframes)} keyframes, "
            f"{len(events)} events, {len(objects)} objects"
        )
        
        # 1. Create chapter group nodes
        if chapter_groups:
            group_count, group_errors = await self._create_chapter_group_nodes(chapter_groups)
            result.chapter_group_nodes_created = group_count
            result.errors.extend(group_errors)
        
        # 2. Create chapter nodes
        if chapters:
            chapter_count, chapter_errors = await self._create_chapter_nodes(chapters)
            result.chapter_nodes_created = chapter_count
            result.errors.extend(chapter_errors)
        
        # 3. Create transcript nodes
        if transcripts:
            transcript_count, transcript_errors = await self._create_transcript_nodes(transcripts)
            result.transcript_nodes_created = transcript_count
            result.errors.extend(transcript_errors)
        
        # 4. Create keyframe nodes
        if keyframes:
            keyframe_count, keyframe_errors = await self._create_keyframe_nodes(keyframes)
            result.keyframe_nodes_created = keyframe_count
            result.errors.extend(keyframe_errors)
        
        # 5. Create event nodes
        event_count, event_errors = await self._create_event_nodes(events)
        result.event_nodes_created = event_count
        result.errors.extend(event_errors)
        
        # 6. Create object nodes
        object_count, object_errors = await self._create_object_nodes(objects)
        result.object_nodes_created = object_count
        result.errors.extend(object_errors)
        
        # 7. Create chapter group temporal edges (NEXT_GROUP/PREV_GROUP)
        if len(chapter_groups) > 1:
            group_temp_count, group_temp_errors = await self._create_group_temporal_edges(chapter_groups)
            result.group_temporal_edges_created = group_temp_count
            result.errors.extend(group_temp_errors)
        
        # 8. Create chapter temporal edges (NEXT_CHAPTER/PREV_CHAPTER)
        if len(chapters) > 1:
            chapter_temp_count, chapter_temp_errors = await self._create_chapter_temporal_edges(chapters)
            result.chapter_temporal_edges_created = chapter_temp_count
            result.errors.extend(chapter_temp_errors)
        
        # 9. Create transcript temporal edges (NEXT_TRANSCRIPT/PREV_TRANSCRIPT)
        if len(transcripts) > 1:
            transcript_temp_count, transcript_temp_errors = await self._create_transcript_temporal_edges(transcripts)
            result.transcript_temporal_edges_created = transcript_temp_count
            result.errors.extend(transcript_temp_errors)
        
        # 10. Create event temporal edges (NEXT_EVENT/PREV_EVENT)
        temporal_count, temporal_errors = await self._create_event_temporal_edges(events)
        result.event_temporal_edges_created = temporal_count
        result.errors.extend(temporal_errors)
        
        # 11. Create hierarchy edges (HAS_CHAPTER, IN_GROUP, HAS_EVENT, IN_CHAPTER)
        hierarchy_count, hierarchy_errors = await self._create_hierarchy_edges(
            chapter_groups, chapters, events, video_id
        )
        result.hierarchy_edges_created = hierarchy_count
        result.errors.extend(hierarchy_errors)
        
        # 12. Create keyframe hierarchy edges (HAS_KEYFRAME)
        if keyframes:
            keyframe_edge_count, keyframe_edge_errors = await self._create_keyframe_edges(keyframes)
            result.keyframe_edges_created = keyframe_edge_count
            result.errors.extend(keyframe_edge_errors)
        
        # 13. Create transcript hierarchy edges (HAS_TRANSCRIPT)
        if transcripts and chapters:
            has_transcript_count, has_transcript_errors = await self._create_transcript_edges(
                transcripts, chapters
            )
            result.has_transcript_edges_created = has_transcript_count
            result.errors.extend(has_transcript_errors)
        
        # 14. Build object lookup for CONTAINS edges
        object_lookup = self._build_object_lookup(objects)
        
        # 15. Create CONTAINS edges (Event → Object)
        contains_count, edge_errors = await self._create_event_object_edges(
            events, object_lookup
        )
        result.contains_edges_created = contains_count
        result.errors.extend(edge_errors)
        
        logger.info(
            f"Graph build complete: "
            f"{result.chapter_group_nodes_created} groups, {result.chapter_nodes_created} chapters, "
            f"{result.transcript_nodes_created} transcripts, {result.event_nodes_created} events, "
            f"{result.object_nodes_created} objects, {result.hierarchy_edges_created} hierarchy, "
            f"{result.event_temporal_edges_created} temporal edges"
        )
        
        return result
    
    async def _create_chapter_group_nodes(
        self,
        groups: List[GraphChapterGroup],
    ) -> Tuple[int, List[str]]:
        """Create chapter group nodes in the graph.
        
        Args:
            groups: List of GraphChapterGroup instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not groups:
            return 0, []
        
        nodes_to_create = []
        for group in groups:
            if not group.id:
                continue
            
            properties = {
                "name": group.name or "",
                "video_id": group.video_id or "",
                "order": group.order or 0,
                "start_time": group.start_time or 0.0,
                "end_time": group.end_time or 0.0,
                "video_duration": group.video_duration or 0.0,
                "summary": group.summary or "",
                "topics": group.topics or [],
                "chapter_indices": group.chapter_indices or [],
            }
            
            if group.embedding_vector:
                properties["embedding_vector"] = group.embedding_vector
            if group.metadata:
                properties["metadata"] = group.metadata
            
            nodes_to_create.append({
                "id": group.id,
                "type": self.NODE_TYPE_CHAPTER_GROUP,
                "properties": properties,
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(nodes_to_create), self.batch_size):
            batch = nodes_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_nodes(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} chapter group nodes in batch"
                    )
            except Exception as e:
                errors.append(f"Chapter group node batch error: {str(e)}")
                logger.error(f"Failed to create chapter group nodes batch: {e}")
        
        return success_count, errors
    
    async def _create_chapter_nodes(
        self,
        chapters: List[GraphChapter],
    ) -> Tuple[int, List[str]]:
        """Create chapter nodes in the graph.
        
        Args:
            chapters: List of GraphChapter instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not chapters:
            return 0, []
        
        nodes_to_create = []
        for chapter in chapters:
            if not chapter.id:
                continue
            
            properties = {
                "video_id": chapter.video_id or "",
                "chunk_index": chapter.chunk_index or 0,
                "start_time": chapter.start_time or 0.0,
                "end_time": chapter.end_time or 0.0,
                "video_duration": chapter.video_duration or 0.0,
                "summary": chapter.summary or "",
                "transcript": chapter.transcript or "",
                "group_index": chapter.group_index,
            }
            
            if chapter.embedding_vector:
                properties["embedding_vector"] = chapter.embedding_vector
            if chapter.metadata:
                properties["metadata"] = chapter.metadata
            
            nodes_to_create.append({
                "id": chapter.id,
                "type": self.NODE_TYPE_CHAPTER,
                "properties": properties,
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(nodes_to_create), self.batch_size):
            batch = nodes_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_nodes(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} chapter nodes in batch"
                    )
            except Exception as e:
                errors.append(f"Chapter node batch error: {str(e)}")
                logger.error(f"Failed to create chapter nodes batch: {e}")
        
        return success_count, errors
    
    async def _create_keyframe_nodes(
        self,
        keyframes: List[GraphKeyframe],
    ) -> Tuple[int, List[str]]:
        """Create keyframe nodes in the graph.
        
        Args:
            keyframes: List of GraphKeyframe instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not keyframes:
            return 0, []
        
        nodes_to_create = []
        for keyframe in keyframes:
            if not keyframe.id:
                continue
            
            properties = {
                "video_id": keyframe.video_id or "",
                "chapter_id": keyframe.chapter_id or "",
                "timestamp": keyframe.timestamp or 0.0,
                "frame_index": keyframe.frame_index or 0,
                "filepath": keyframe.filepath or "",
                "blob_name": keyframe.blob_name or "",
                "blob_url": keyframe.blob_url or "",
                "motion_score": keyframe.motion_score or 0.0,
            }
            
            if keyframe.embedding_vector:
                properties["embedding_vector"] = keyframe.embedding_vector
            if keyframe.metadata:
                properties["metadata"] = keyframe.metadata
            
            nodes_to_create.append({
                "id": keyframe.id,
                "type": self.NODE_TYPE_KEYFRAME,
                "properties": properties,
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(nodes_to_create), self.batch_size):
            batch = nodes_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_nodes(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} keyframe nodes in batch"
                    )
            except Exception as e:
                errors.append(f"Keyframe node batch error: {str(e)}")
                logger.error(f"Failed to create keyframe nodes batch: {e}")
        
        return success_count, errors
    
    async def _create_transcript_nodes(
        self,
        transcripts: List[GraphTranscript],
    ) -> Tuple[int, List[str]]:
        """Create transcript nodes in the graph.
        
        Transcript nodes contain raw verbal content separate from the Chapter's
        multimodal summary. This enables targeted retrieval for verbal-only queries.
        
        Args:
            transcripts: List of GraphTranscript instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not transcripts:
            return 0, []
        
        nodes_to_create = []
        for transcript in transcripts:
            if not transcript.id:
                continue
            
            properties = {
                "video_id": transcript.video_id or "",
                "chunk_index": transcript.chunk_index or 0,
                "transcript": transcript.transcript or "",
                "start_time": transcript.start_time or 0.0,
                "end_time": transcript.end_time or 0.0,
                "video_duration": transcript.video_duration or 0.0,
            }
            
            if transcript.embedding_vector:
                properties["embedding_vector"] = transcript.embedding_vector
            if transcript.metadata:
                properties["metadata"] = transcript.metadata
            
            nodes_to_create.append({
                "id": transcript.id,
                "type": self.NODE_TYPE_TRANSCRIPT,
                "properties": properties,
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(nodes_to_create), self.batch_size):
            batch = nodes_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_nodes(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} transcript nodes in batch"
                    )
            except Exception as e:
                errors.append(f"Transcript node batch error: {str(e)}")
                logger.error(f"Failed to create transcript nodes batch: {e}")
        
        return success_count, errors
    
    async def _create_event_nodes(
        self,
        events: List[GraphEvent],
    ) -> Tuple[int, List[str]]:
        """Create event nodes in the graph.
        
        Args:
            events: List of GraphEvent instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not events:
            return 0, []
        
        nodes_to_create = []
        for event in events:
            if not event.id:
                continue
            
            properties = {
                "description": event.description or "",
                "video_id": event.video_id or "",
                "timestamp": event.timestamp or 0.0,
                "duration": event.duration or 0.0,
                "event_type": event.event_type or "action",
                "participants": event.participants or [],
                "chapter_index": event.chapter_index,
                "sequence_number": event.sequence_number,
            }
            
            if event.metadata:
                properties["metadata"] = event.metadata
            
            nodes_to_create.append({
                "id": event.id,
                "type": self.NODE_TYPE_EVENT,
                "properties": properties,
            })
        
        # Batch create nodes
        errors = []
        success_count = 0
        
        for i in range(0, len(nodes_to_create), self.batch_size):
            batch = nodes_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_nodes(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} event nodes in batch {i // self.batch_size}"
                    )
            except Exception as e:
                errors.append(f"Event node batch error: {str(e)}")
                logger.error(f"Failed to create event nodes batch: {e}")
        
        return success_count, errors
    
    async def _create_object_nodes(
        self,
        objects: List[GraphObject],
    ) -> Tuple[int, List[str]]:
        """Create object nodes in the graph.
        
        Args:
            objects: List of GraphObject instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not objects:
            return 0, []
        
        nodes_to_create = []
        for obj in objects:
            if not obj.id:
                continue
            
            properties = {
                "name": obj.name or "",
                "video_id": obj.video_id or "",
                "first_seen": obj.first_seen or 0.0,
                "last_seen": obj.last_seen or 0.0,
                "object_type": obj.object_type or "item",
                "appearance": obj.appearance or [],
                "identity": obj.identity or [],
                "is_canonical": obj.is_canonical,
                "appearance_count": obj.appearance_count,
            }
            
            if obj.merged_from:
                properties["merged_from"] = obj.merged_from
            
            if obj.metadata:
                properties["metadata"] = obj.metadata
            
            nodes_to_create.append({
                "id": obj.id,
                "type": self.NODE_TYPE_OBJECT,
                "properties": properties,
            })
        
        # Batch create nodes
        errors = []
        success_count = 0
        
        for i in range(0, len(nodes_to_create), self.batch_size):
            batch = nodes_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_nodes(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} object nodes in batch {i // self.batch_size}"
                    )
            except Exception as e:
                errors.append(f"Object node batch error: {str(e)}")
                logger.error(f"Failed to create object nodes batch: {e}")
        
        return success_count, errors
    
    def _build_object_lookup(
        self,
        objects: List[GraphObject],
    ) -> Dict[str, str]:
        """Build lookup map from object name to object ID.
        
        Args:
            objects: List of GraphObject instances.
            
        Returns:
            Dictionary mapping normalized object names to IDs.
        """
        lookup = {}
        for obj in objects:
            if obj.name and obj.id:
                normalized = obj.name.lower().strip()
                lookup[normalized] = obj.id
        return lookup
    
    async def _create_keyframe_edges(
        self,
        keyframes: List[GraphKeyframe],
    ) -> Tuple[int, List[str]]:
        """Create hierarchy edges between chapters and keyframes.
        
        Creates single-direction edge (query code handles bidirectional traversal):
        - HAS_KEYFRAME: Chapter → Keyframe
        
        Args:
            keyframes: List of GraphKeyframe instances.
            
        Returns:
            Tuple of (edge_count, error_messages).
        """
        if not keyframes:
            return 0, []
        
        edges_to_create = []
        
        for keyframe in keyframes:
            if not keyframe.id or not keyframe.chapter_id:
                continue
            
            # HAS_KEYFRAME: Chapter → Keyframe
            edges_to_create.append({
                "source_id": keyframe.chapter_id,
                "target_id": keyframe.id,
                "type": self.EDGE_HAS_KEYFRAME,
                "properties": {
                    "timestamp": keyframe.timestamp or 0.0,
                    "frame_index": keyframe.frame_index or 0,
                },
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(
                        f"Failed to create {result['failed']} keyframe edges in batch"
                    )
            except Exception as e:
                errors.append(f"Keyframe edge batch error: {str(e)}")
                logger.error(f"Failed to create keyframe edges batch: {e}")
        
        return success_count, errors
    
    async def _create_event_object_edges(
        self,
        events: List[GraphEvent],
        object_lookup: Dict[str, str],
    ) -> Tuple[int, List[str]]:
        """Create edges between events and objects.
        
        Creates single-direction edge (query code handles bidirectional traversal):
        - CONTAINS: Event → Object (event contains/involves this object)
        
        Uses linked_object_ids from event metadata (set by temporal_graph step).
        Falls back to participant name matching if metadata not available.
        
        Args:
            events: List of GraphEvent instances.
            object_lookup: Map of object names to IDs.
            
        Returns:
            Tuple of (contains_count, error_messages).
        """
        if not events:
            return 0, []
        
        contains_edges = []
        
        for event in events:
            if not event.id:
                continue
            
            linked_ids = []
            
            # Primary: Use linked_object_ids from metadata (set by temporal_graph)
            if event.metadata and "linked_object_ids" in event.metadata:
                linked_ids = list(event.metadata["linked_object_ids"])
            
            # Fallback: Match participants to objects by name
            elif event.participants and object_lookup:
                for participant in event.participants:
                    normalized = participant.lower().strip()
                    if normalized in object_lookup:
                        obj_id = object_lookup[normalized]
                        if obj_id not in linked_ids:
                            linked_ids.append(obj_id)
                    else:
                        # Fuzzy match
                        for obj_name, obj_id in object_lookup.items():
                            if obj_name in normalized or normalized in obj_name:
                                if obj_id not in linked_ids:
                                    linked_ids.append(obj_id)
                                break
            
            # Create CONTAINS edges for each linked object
            for obj_id in linked_ids:
                # CONTAINS: Event → Object
                contains_edges.append({
                    "source_id": event.id,
                    "target_id": obj_id,
                    "type": self.EDGE_CONTAINS,
                    "properties": {
                        "video_id": event.video_id,
                        "event_timestamp": event.timestamp,
                    },
                })
        
        errors = []
        contains_count = 0
        
        # Batch create CONTAINS edges
        for i in range(0, len(contains_edges), self.batch_size):
            batch = contains_edges[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                contains_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} CONTAINS edges")
            except Exception as e:
                errors.append(f"CONTAINS edge batch error: {str(e)}")
                logger.error(f"Failed to create CONTAINS edges batch: {e}")
        
        return contains_count, errors
    
    async def _create_event_temporal_edges(
        self,
        events: List[GraphEvent],
    ) -> Tuple[int, List[str]]:
        """Create temporal sequence edges (NEXT_EVENT/PREV_EVENT) between events.
        
        Orders events by timestamp and creates bidirectional temporal links.
        
        Args:
            events: List of GraphEvent instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if len(events) < 2:
            return 0, []
        
        # Log filtering info for debugging
        total_events = len(events)
        valid_events = [e for e in events if e.id and e.timestamp is not None]
        filtered_out = total_events - len(valid_events)
        
        if filtered_out > 0:
            logger.warning(
                f"Temporal edges: filtered out {filtered_out}/{total_events} events "
                f"(missing id or timestamp)"
            )
            # Log samples of filtered events for debugging
            for e in events[:5]:
                if not e.id or e.timestamp is None:
                    logger.debug(f"  Filtered event: id={e.id}, timestamp={e.timestamp}")
        
        # Sort events by timestamp, then sequence number
        sorted_events = sorted(
            valid_events,
            key=lambda e: (e.timestamp or 0.0, e.sequence_number or 0),
        )
        
        logger.info(f"Creating temporal edges for {len(sorted_events)} events (from {total_events} total)")
        
        if len(sorted_events) < 2:
            return 0, []
        
        edges_to_create = []
        
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            time_gap = (next_event.timestamp or 0.0) - (current_event.timestamp or 0.0)
            
            # NEXT edge: current → next
            edges_to_create.append({
                "source_id": current_event.id,
                "target_id": next_event.id,
                "type": self.EDGE_NEXT_EVENT,
                "properties": {
                    "video_id": current_event.video_id,
                    "time_gap_seconds": time_gap,
                    "sequence_from": current_event.sequence_number,
                    "sequence_to": next_event.sequence_number,
                },
            })
            
            # PREVIOUS edge: next → current
            edges_to_create.append({
                "source_id": next_event.id,
                "target_id": current_event.id,
                "type": self.EDGE_PREV_EVENT,
                "properties": {
                    "video_id": current_event.video_id,
                    "time_gap_seconds": time_gap,
                    "sequence_from": next_event.sequence_number,
                    "sequence_to": current_event.sequence_number,
                },
            })
        
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
                        f"Failed to create {result['failed']} temporal edges in batch {i // self.batch_size}"
                    )
            except Exception as e:
                errors.append(f"Temporal edge batch error: {str(e)}")
                logger.error(f"Failed to create temporal edges batch: {e}")
        
        return success_count, errors
    
    async def _create_group_temporal_edges(
        self,
        groups: List[GraphChapterGroup],
    ) -> Tuple[int, List[str]]:
        """Create temporal sequence edges between chapter groups.
        
        Creates NEXT_GROUP and PREV_GROUP edges based on group order.
        
        Args:
            groups: List of GraphChapterGroup instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if len(groups) < 2:
            return 0, []
        
        # Sort by order
        sorted_groups = sorted(
            [g for g in groups if g.id and g.order is not None],
            key=lambda g: g.order or 0
        )
        
        if len(sorted_groups) < 2:
            return 0, []
        
        edges_to_create = []
        
        for i in range(len(sorted_groups) - 1):
            current_group = sorted_groups[i]
            next_group = sorted_groups[i + 1]
            
            # NEXT_GROUP: current → next
            edges_to_create.append({
                "source_id": current_group.id,
                "target_id": next_group.id,
                "type": self.EDGE_NEXT_GROUP,
                "properties": {
                    "video_id": current_group.video_id,
                    "order_from": current_group.order,
                    "order_to": next_group.order,
                },
            })
            
            # PREV_GROUP: next → current
            edges_to_create.append({
                "source_id": next_group.id,
                "target_id": current_group.id,
                "type": self.EDGE_PREV_GROUP,
                "properties": {
                    "video_id": current_group.video_id,
                    "order_from": next_group.order,
                    "order_to": current_group.order,
                },
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} group temporal edges")
            except Exception as e:
                errors.append(f"Group temporal edge batch error: {str(e)}")
                logger.error(f"Failed to create group temporal edges batch: {e}")
        
        return success_count, errors
    
    async def _create_chapter_temporal_edges(
        self,
        chapters: List[GraphChapter],
    ) -> Tuple[int, List[str]]:
        """Create temporal sequence edges between chapters.
        
        Creates NEXT_CHAPTER and PREV_CHAPTER edges based on chunk_index.
        
        Args:
            chapters: List of GraphChapter instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if len(chapters) < 2:
            return 0, []
        
        # Sort by chunk_index
        sorted_chapters = sorted(
            [c for c in chapters if c.id and c.chunk_index is not None],
            key=lambda c: c.chunk_index or 0
        )
        
        if len(sorted_chapters) < 2:
            return 0, []
        
        edges_to_create = []
        
        for i in range(len(sorted_chapters) - 1):
            current_chapter = sorted_chapters[i]
            next_chapter = sorted_chapters[i + 1]
            
            time_gap = (next_chapter.start_time or 0.0) - (current_chapter.end_time or 0.0)
            
            # NEXT_CHAPTER: current → next
            edges_to_create.append({
                "source_id": current_chapter.id,
                "target_id": next_chapter.id,
                "type": self.EDGE_NEXT_CHAPTER,
                "properties": {
                    "video_id": current_chapter.video_id,
                    "chunk_index_from": current_chapter.chunk_index,
                    "chunk_index_to": next_chapter.chunk_index,
                    "time_gap_seconds": time_gap,
                },
            })
            
            # PREV_CHAPTER: next → current
            edges_to_create.append({
                "source_id": next_chapter.id,
                "target_id": current_chapter.id,
                "type": self.EDGE_PREV_CHAPTER,
                "properties": {
                    "video_id": current_chapter.video_id,
                    "chunk_index_from": next_chapter.chunk_index,
                    "chunk_index_to": current_chapter.chunk_index,
                    "time_gap_seconds": time_gap,
                },
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} chapter temporal edges")
            except Exception as e:
                errors.append(f"Chapter temporal edge batch error: {str(e)}")
                logger.error(f"Failed to create chapter temporal edges batch: {e}")
        
        return success_count, errors
    
    async def _create_transcript_temporal_edges(
        self,
        transcripts: List[GraphTranscript],
    ) -> Tuple[int, List[str]]:
        """Create temporal sequence edges between transcripts.
        
        Creates NEXT_TRANSCRIPT and PREV_TRANSCRIPT edges based on chunk_index.
        
        Args:
            transcripts: List of GraphTranscript instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if len(transcripts) < 2:
            return 0, []
        
        # Sort by chunk_index
        sorted_transcripts = sorted(
            [t for t in transcripts if t.id and t.chunk_index is not None],
            key=lambda t: t.chunk_index or 0
        )
        
        if len(sorted_transcripts) < 2:
            return 0, []
        
        edges_to_create = []
        
        for i in range(len(sorted_transcripts) - 1):
            current_transcript = sorted_transcripts[i]
            next_transcript = sorted_transcripts[i + 1]
            
            time_gap = (next_transcript.start_time or 0.0) - (current_transcript.end_time or 0.0)
            
            # NEXT_TRANSCRIPT: current → next
            edges_to_create.append({
                "source_id": current_transcript.id,
                "target_id": next_transcript.id,
                "type": self.EDGE_NEXT_TRANSCRIPT,
                "properties": {
                    "video_id": current_transcript.video_id,
                    "chunk_index_from": current_transcript.chunk_index,
                    "chunk_index_to": next_transcript.chunk_index,
                    "time_gap_seconds": time_gap,
                },
            })
            
            # PREV_TRANSCRIPT: next → current
            edges_to_create.append({
                "source_id": next_transcript.id,
                "target_id": current_transcript.id,
                "type": self.EDGE_PREV_TRANSCRIPT,
                "properties": {
                    "video_id": current_transcript.video_id,
                    "chunk_index_from": next_transcript.chunk_index,
                    "chunk_index_to": current_transcript.chunk_index,
                    "time_gap_seconds": time_gap,
                },
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} transcript temporal edges")
            except Exception as e:
                errors.append(f"Transcript temporal edge batch error: {str(e)}")
                logger.error(f"Failed to create transcript temporal edges batch: {e}")
        
        return success_count, errors
    
    async def _create_transcript_edges(
        self,
        transcripts: List[GraphTranscript],
        chapters: List[GraphChapter],
    ) -> Tuple[int, List[str]]:
        """Create HAS_TRANSCRIPT edges from Chapter to Transcript nodes.
        
        Links each Chapter to its corresponding Transcript (1:1 mapping via chunk_index).
        
        Args:
            transcripts: List of GraphTranscript instances.
            chapters: List of GraphChapter instances.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        if not transcripts or not chapters:
            return 0, []
        
        # Build chapter lookup by chunk_index
        chapter_by_chunk = {c.chunk_index: c for c in chapters if c.id and c.chunk_index is not None}
        
        edges_to_create = []
        
        for transcript in transcripts:
            if not transcript.id or transcript.chunk_index is None:
                continue
            
            chapter = chapter_by_chunk.get(transcript.chunk_index)
            if not chapter:
                continue
            
            # HAS_TRANSCRIPT: Chapter → Transcript
            edges_to_create.append({
                "source_id": chapter.id,
                "target_id": transcript.id,
                "type": self.EDGE_HAS_TRANSCRIPT,
                "properties": {
                    "video_id": transcript.video_id,
                    "chunk_index": transcript.chunk_index,
                },
            })
        
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} HAS_TRANSCRIPT edges")
            except Exception as e:
                errors.append(f"HAS_TRANSCRIPT edge batch error: {str(e)}")
                logger.error(f"Failed to create HAS_TRANSCRIPT edges batch: {e}")
        
        return success_count, errors
    
    async def _create_hierarchy_edges(
        self,
        groups: List[GraphChapterGroup],
        chapters: List[GraphChapter],
        events: List[GraphEvent],
        video_id: str,
    ) -> Tuple[int, List[str]]:
        """Create hierarchy edges connecting groups, chapters, and events.
        
        Creates single-direction containment edges (query code handles bidirectional traversal):
        - HAS_CHAPTER: ChapterGroup → Chapter
        - HAS_EVENT: Chapter → Event
        
        Args:
            groups: List of GraphChapterGroup instances.
            chapters: List of GraphChapter instances.
            events: List of GraphEvent instances.
            video_id: Video identifier.
            
        Returns:
            Tuple of (success_count, error_messages).
        """
        edges_to_create = []
        
        # Build chapter lookup by chunk_index
        chapter_by_index: Dict[int, GraphChapter] = {}
        for chapter in chapters:
            if chapter.chunk_index is not None and chapter.id:
                chapter_by_index[chapter.chunk_index] = chapter
        
        # 1. Create HAS_CHAPTER edges (ChapterGroup → Chapter)
        for group in groups:
            if not group.id or not group.chapter_indices:
                continue
            
            for chapter_idx in group.chapter_indices:
                chapter = chapter_by_index.get(chapter_idx)
                if not chapter or not chapter.id:
                    continue
                
                # HAS_CHAPTER: Group → Chapter
                edges_to_create.append({
                    "source_id": group.id,
                    "target_id": chapter.id,
                    "type": self.EDGE_HAS_CHAPTER,
                    "properties": {
                        "video_id": video_id,
                        "group_order": group.order,
                        "chapter_index": chapter_idx,
                    },
                })
        
        # 2. Create HAS_EVENT edges (Chapter → Event)
        for event in events:
            if not event.id or event.chapter_index is None:
                continue
            
            chapter = chapter_by_index.get(event.chapter_index)
            if not chapter or not chapter.id:
                continue
            
            # HAS_EVENT: Chapter → Event
            edges_to_create.append({
                "source_id": chapter.id,
                "target_id": event.id,
                "type": self.EDGE_HAS_EVENT,
                "properties": {
                    "video_id": video_id,
                    "chapter_index": event.chapter_index,
                    "event_timestamp": event.timestamp,
                },
            })
        
        if not edges_to_create:
            return 0, []
        
        errors = []
        success_count = 0
        
        for i in range(0, len(edges_to_create), self.batch_size):
            batch = edges_to_create[i:i + self.batch_size]
            try:
                result = await self.graph_provider.batch_create_edges(batch)
                success_count += result.get("success", 0)
                if result.get("failed", 0) > 0:
                    errors.append(f"Failed to create {result['failed']} hierarchy edges")
            except Exception as e:
                errors.append(f"Hierarchy edge batch error: {str(e)}")
                logger.error(f"Failed to create hierarchy edges batch: {e}")
        
        return success_count, errors
