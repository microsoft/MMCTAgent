"""Temporal Graph pipeline step for event and object extraction.

This step extracts events and objects from video chapters to build a temporal
knowledge graph for advanced video retrieval.

Architecture:
1. Parallel chapter processing with configurable concurrency
2. Per-chapter: Events + objects extracted simultaneously
3. Per-chapter: Objects linked to events via timestamp overlap
4. Post-processing: Batch object deduplication using embeddings
5. Post-processing: Event links updated to canonical object IDs

Embedding Models (fastembed, CPU-optimized):
- Events: BAAI/bge-small-en-v1.5 (384-dim)
- Objects: snowflake/snowflake-arctic-embed-s (384-dim)

Outputs:
- events: List of GraphEvent with linked_object_ids in metadata
- objects: Deduplicated list of GraphObject with linked_event_ids in metadata
"""

import asyncio
from typing import List, Dict, Any, Tuple

from loguru import logger

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.video_pipeline.core.ingestion.models import GraphEvent, GraphObject

from .config import (
    MAX_EVENTS_PER_CHAPTER,
    MAX_OBJECTS_PER_EVENT,
    MIN_EVENT_DURATION_MS,
    MAX_EXTRACTION_RETRIES,
    MAX_CONCURRENT_EXTRACTIONS,
)
from .event_extractor import EventExtractor
from .object_extractor import ObjectExtractor
from .local_embeddings import (
    get_event_embedding_provider,
    get_object_embedding_provider,
)


@register_step("ingestion.temporal_graph")
class TemporalGraphStep(PipelineStep):
    """Pipeline step for temporal graph construction.
    
    Extracts events and objects from enriched chapters to build
    a temporal knowledge graph for advanced video retrieval.
    
    Uses fastembed with CPU execution for local embedding generation:
    - Events: BAAI/bge-small-en-v1.5 (384-dim)
    - Objects: snowflake/snowflake-arctic-embed-s (384-dim)
    
    Extraction Architecture:
    - Parallel processing: Events + objects extracted simultaneously per chapter
    - Timestamp-based linking: Objects linked to events via time overlap
    - Batch deduplication: Objects deduplicated after all chapters processed
    
    Features:
    - Atomic event extraction from chapter content (multimodal: frames + transcript)
    - Visual object extraction from keyframes (visual-only)
    - Cross-chapter object deduplication using embeddings
    - Timestamp-based event-object linking (no fuzzy name matching)
    - Configurable concurrency for parallel chapter processing
    
    Params:
        source_chapters_step: Step ID for chapters (default: "dense_chapters")
        source_keyframes_step: Step ID for keyframes (default: "dense_keyframes")
        max_events_per_chapter: Maximum events per chapter (default: 10)
        max_objects_per_chapter: Maximum objects per chapter (default: 15)
        min_event_duration_ms: Minimum event duration in ms (default: 500)
        generate_embeddings: Whether to generate embeddings (default: True)
        enable_deduplication: Whether to deduplicate objects (default: True)
        embedding_device: Device for embeddings - unused, always CPU (default: "cpu")
        max_frames_per_chapter: Max frames per chapter for extraction (default: 12)
        max_concurrent_chapters: Max chapters to process in parallel (default: 4)
    """
    
    step_type = "ingestion.temporal_graph"
    description = "Extract temporal events and objects for knowledge graph."
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute temporal graph extraction with parallel processing.
        
        Architecture:
        1. For each chapter (with configurable concurrency):
           - Extract events and objects in parallel
           - Link objects to events via timestamp overlap
        2. After all chapters complete:
           - Batch deduplicate objects using embeddings
           - Update event links to point to canonical object IDs
        
        Args:
            context: Pipeline step context with data store and providers
            
        Returns:
            StepResult containing extracted events and deduplicated objects
        """
        source_step: str = self.get_param(
            "source_chapters_step", context, default="dense_chapters"
        )
        chapters: List[Dict[str, Any]] = (
            context.data_store.get(source_step, "raw_chapters") 
            or context.data_store.get(source_step, "chapters")
            or []
        )
        
        if not chapters:
            chapters = context.data_store.get("chapter_enrichment", "raw_chapters") or []
        
        if not chapters:
            chapters = context.data_store.get("chapters", "raw_chapters") or []
        
        if not chapters:
            logger.warning("No chapters found for temporal graph extraction")
            return StepResult(
                step_id=self.step_id,
                outputs={
                    "events": [],
                    "objects": [],
                    "event_count": 0,
                    "object_count": 0,
                },
                metrics={
                    "events_extracted": 0,
                    "objects_extracted": 0,
                    "chapters_processed": 0,
                },
            )
        
        video_id: str = getattr(context, "video_id", "")
        logger.info(
            f"Starting temporal graph extraction for video {video_id} "
            f"with {len(chapters)} chapters"
        )
        
        # Get keyframes data for visual extraction
        source_keyframes_step = self.get_param(
            "source_keyframes_step", context, default="dense_keyframes"
        )
        keyframes_data: List[Dict[str, Any]] = (
            context.data_store.get(source_keyframes_step, "keyframes_per_chunk")
            or context.data_store.get("keyframes", "keyframes_per_chunk")
            or []
        )
        
        # Get configuration parameters
        max_events = self.get_param(
            "max_events_per_chapter", context, default=MAX_EVENTS_PER_CHAPTER
        )
        max_objects = self.get_param(
            "max_objects_per_chapter", context, default=15
        )
        min_duration = self.get_param(
            "min_event_duration_ms", context, default=MIN_EVENT_DURATION_MS
        )
        generate_embeddings = self.get_param(
            "generate_embeddings", context, default=True
        )
        enable_dedup = self.get_param(
            "enable_deduplication", context, default=True
        )
        embedding_device = self.get_param(
            "embedding_device", context, default="cpu"
        )
        max_frames_per_chapter = self.get_param(
            "max_frames_per_chapter", context, default=12
        )
        max_concurrent = self.get_param(
            "max_concurrent_chapters", context, default=MAX_CONCURRENT_EXTRACTIONS
        )
        
        # Get separate local embedding providers for events and objects
        event_embedding_provider = None
        object_embedding_provider = None
        if generate_embeddings:
            try:
                event_embedding_provider = get_event_embedding_provider()
                object_embedding_provider = get_object_embedding_provider()
                logger.info("Using local embedding providers (BGE for events, Arctic for objects)")
            except ImportError as e:
                logger.warning(f"Local embeddings unavailable: {e}. Skipping embeddings.")
        
        event_extractor = EventExtractor(
            llm_provider=context.provider.llm_provider,
            embedding_provider=event_embedding_provider,
            max_events_per_chapter=max_events,
            min_event_duration_ms=min_duration,
            max_retries=MAX_EXTRACTION_RETRIES,
            max_frames_per_chapter=max_frames_per_chapter,
        )
        
        object_extractor = ObjectExtractor(
            llm_provider=context.provider.llm_provider,
            embedding_provider=object_embedding_provider,
            max_objects_per_chapter=max_objects,
            similarity_threshold=0.80 if enable_dedup else 1.0,
            max_retries=MAX_EXTRACTION_RETRIES,
            max_frames_per_chapter=max_frames_per_chapter,
        )
        
        # Process chapters with configurable concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chapter(
            chapter_idx: int,
            chapter: Dict[str, Any],
            keyframes: Dict[str, Any],
        ) -> Tuple[List[GraphEvent], List[GraphObject]]:
            """Process a single chapter: extract events + objects in parallel, then link."""
            async with semaphore:
                logger.info(f"Processing chapter {chapter_idx + 1}/{len(chapters)}")
                
                # Extract events and objects in parallel
                events_task = event_extractor.extract_events_from_chapter(
                    chapter_data=chapter,
                    keyframes=keyframes,
                    chapter_index=chapter_idx,
                    video_id=video_id,
                )
                objects_task = object_extractor.extract_objects_from_chapter(
                    chapter_data=chapter,
                    keyframes=keyframes,
                    chapter_index=chapter_idx,
                    video_id=video_id,
                    skip_deduplication=True,  # Batch dedup later
                )
                
                chapter_events, chapter_objects = await asyncio.gather(
                    events_task, objects_task
                )
                
                # Link objects to events within this chapter (timestamp overlap)
                self._link_objects_to_events_by_timestamp(chapter_events, chapter_objects)
                
                return chapter_events, chapter_objects
        
        # Create tasks for all chapters
        tasks = [
            process_chapter(
                chapter_idx=idx,
                chapter=chapter,
                keyframes=keyframes_data[idx] if idx < len(keyframes_data) else {},
            )
            for idx, chapter in enumerate(chapters)
        ]
        
        # Execute all chapter processing tasks
        results = await asyncio.gather(*tasks)
        
        # Collect all events and objects
        all_events: List[GraphEvent] = []
        all_objects: List[GraphObject] = []
        
        for chapter_events, chapter_objects in results:
            all_events.extend(chapter_events)
            all_objects.extend(chapter_objects)
        
        # Assign global sequence numbers to events
        all_events.sort(key=lambda e: (e.chapter_index or 0, e.timestamp or 0))
        for idx, event in enumerate(all_events, start=1):
            event.sequence_number = idx
        
        # Batch deduplicate objects and update event links
        if enable_dedup and all_objects:
            all_objects, id_mapping = object_extractor.deduplicate_objects_batch(all_objects)
            self._update_event_object_links(all_events, id_mapping)
        
        # Update object metadata with linked events (reverse mapping)
        self._update_object_linked_events(all_events, all_objects)
        
        event_dicts = [event.model_dump() for event in all_events]
        object_dicts = [obj.model_dump() for obj in all_objects]
        
        logger.info(
            f"Temporal graph extraction complete: "
            f"{len(all_events)} events, {len(all_objects)} objects"
        )
        
        return StepResult(
            step_id=self.step_id,
            outputs={
                "events": event_dicts,
                "objects": object_dicts,
                "event_count": len(all_events),
                "object_count": len(all_objects),
                "graph_events": all_events,
                "graph_objects": all_objects,
            },
            metrics={
                "events_extracted": len(all_events),
                "objects_extracted": len(all_objects),
                "chapters_processed": len(chapters),
                "events_per_chapter": len(all_events) / max(len(chapters), 1),
                "objects_per_chapter": len(all_objects) / max(len(chapters), 1),
            },
        )
    
    def _link_objects_to_events_by_timestamp(
        self,
        events: List[GraphEvent],
        objects: List[GraphObject],
    ) -> None:
        """Link objects to events within a chapter based on timestamp overlap.
        
        An object is linked to an event if the object's visible timespan
        [first_seen, last_seen] overlaps with the event's timespan
        [timestamp, timestamp + duration].
        
        This is called per-chapter before deduplication, so object IDs
        are still chapter-specific.
        
        Args:
            events: List of events from a single chapter
            objects: List of objects from the same chapter
        """
        for event in events:
            linked_object_ids: List[str] = []
            event_start = event.timestamp or 0.0
            event_end = event_start + (event.duration or 0.0)
            
            for obj in objects:
                obj_first_seen = obj.first_seen or 0.0
                obj_last_seen = obj.last_seen or obj_first_seen
                
                # Check if object's visible timespan overlaps with event timespan
                if obj_first_seen <= event_end and obj_last_seen >= event_start:
                    linked_object_ids.append(obj.id)
            
            linked_object_ids = linked_object_ids[:MAX_OBJECTS_PER_EVENT]
            
            if event.metadata is None:
                event.metadata = {}
            event.metadata["linked_object_ids"] = linked_object_ids
    
    def _update_event_object_links(
        self,
        events: List[GraphEvent],
        id_mapping: Dict[str, str],
    ) -> None:
        """Update event object links to point to canonical object IDs after dedup.
        
        Args:
            events: All events with linked_object_ids in metadata
            id_mapping: Mapping from original object ID to canonical object ID
        """
        for event in events:
            if not event.metadata or "linked_object_ids" not in event.metadata:
                continue
            
            original_ids = event.metadata["linked_object_ids"]
            canonical_ids = []
            
            for obj_id in original_ids:
                canonical_id = id_mapping.get(obj_id, obj_id)
                if canonical_id not in canonical_ids:
                    canonical_ids.append(canonical_id)
            
            event.metadata["linked_object_ids"] = canonical_ids[:MAX_OBJECTS_PER_EVENT]
    
    def _update_object_linked_events(
        self,
        events: List[GraphEvent],
        objects: List[GraphObject],
    ) -> None:
        """Build reverse mapping: update objects with their linked event IDs.
        
        Called after deduplication to ensure objects have correct event links.
        
        Args:
            events: All events with linked_object_ids pointing to canonical IDs
            objects: Deduplicated objects
        """
        # Build object_id -> event_ids mapping
        event_by_object: Dict[str, List[str]] = {}
        for event in events:
            for obj_id in (event.metadata or {}).get("linked_object_ids", []):
                if obj_id not in event_by_object:
                    event_by_object[obj_id] = []
                event_by_object[obj_id].append(event.id)
        
        # Update each object's metadata
        for obj in objects:
            if obj.metadata is None:
                obj.metadata = {}
            obj.metadata["linked_event_ids"] = event_by_object.get(obj.id, [])
