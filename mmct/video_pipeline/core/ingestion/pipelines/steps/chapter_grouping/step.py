"""Chapter Grouping pipeline step for semantic chapter organization.

Groups consecutive chapters by embedding similarity and temporal proximity,
then generates LLM summaries for each group to enable hierarchical navigation.

Outputs:
- chapter_groups: List of ChapterGroup with indices, time bounds, and summaries
- groups: ChapterGroup model instances for downstream steps
"""

import logging
from typing import List, Dict, Any

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.video_pipeline.core.ingestion.models import ChapterGroup

from .config import (
    CHAPTER_GROUPING_THRESHOLD,
    CHAPTER_TEMPORAL_WINDOW,
)
from .grouper import ChapterGrouper
from .summarizer import GroupSummarizer


logger = logging.getLogger(__name__)


@register_step("ingestion.chapter_grouping")
class ChapterGroupingStep(PipelineStep):
    """Pipeline step for grouping chapters by semantic similarity.
    
    Groups consecutive chapters based on embedding similarity and temporal
    proximity, then generates LLM summaries for each group.
    
    Features:
    - Sliding window grouping with configurable thresholds
    - Temporal proximity constraints
    - LLM-generated group summaries
    - Group embedding computation
    
    Params:
        source_chapters_step: Step ID for source chapters (default: "chapters")
        similarity_threshold: Cosine similarity threshold for grouping (default: 0.7)
        temporal_window: Maximum chapter distance for grouping (default: 5)
        generate_summaries: Whether to generate LLM summaries (default: True)
    """
    
    step_type = "ingestion.chapter_grouping"
    description = "Group chapters by semantic similarity and temporal proximity."
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute chapter grouping and summarization.
        
        Args:
            context: Pipeline step context with data store and providers
            
        Returns:
            StepResult containing chapter groups and metrics
        """
        # Get source chapters with fallback chain
        source_step: str = self.get_param(
            "source_chapters_step", context, default="chapters"
        )
        chapters: List[Dict[str, Any]] = (
            context.data_store.get(source_step, "raw_chapters")
            or context.data_store.get(source_step, "chapters")
            or context.data_store.get("dense_chapters", "raw_chapters")
            or context.data_store.get("dense_chapters", "chapters")
            or []
        )
        
        if not chapters:
            chapters = context.data_store.get("chapter_enrichment", "raw_chapters") or []
        
        if not chapters:
            chapters = context.data_store.get("chapters", "raw_chapters") or []
        
        if not chapters:
            logger.warning("No chapters found for chapter grouping")
            return StepResult(
                step_id=self.step_id,
                outputs={
                    "chapter_groups": [],
                    "group_count": 0,
                },
                metrics={
                    "groups_created": 0,
                    "chapters_processed": 0,
                    "avg_chapters_per_group": 0.0,
                },
            )
        
        video_id: str = getattr(context, "video_id", "")
        video_duration: float = getattr(context, "video_duration", 0.0)
        logger.info(
            f"Starting chapter grouping for video {video_id} "
            f"with {len(chapters)} chapters"
        )
        
        # Get configuration parameters
        similarity_threshold = self.get_param(
            "similarity_threshold", context, default=CHAPTER_GROUPING_THRESHOLD
        )
        temporal_window = self.get_param(
            "temporal_window", context, default=CHAPTER_TEMPORAL_WINDOW
        )
        generate_summaries = self.get_param(
            "generate_summaries", context, default=True
        )
        
        # Get optional playlist metadata from context
        playlist_id = self.get_param("playlist_id", context, default=None)
        playlist_order = self.get_param("playlist_order", context, default=None)
        
        # Initialize grouper
        grouper = ChapterGrouper(
            similarity_threshold=similarity_threshold,
            temporal_window=temporal_window,
        )
        
        # Group chapters
        groups, messages = grouper.group_chapters(
            chapters=chapters,
            video_id=video_id,
            video_duration=video_duration,
            playlist_id=playlist_id,
            playlist_order=playlist_order,
        )
        
        # Log grouping messages
        for msg in messages:
            logger.info(msg)
        
        # Generate summaries if enabled
        if generate_summaries and groups and hasattr(context.provider, "llm_provider"):
            logger.info(f"Generating summaries for {len(groups)} chapter groups")
            
            summarizer = GroupSummarizer()
            groups = await summarizer.generate_group_summaries(
                groups=groups,
                chapters=chapters,
                llm_provider=context.provider.llm_provider,
            )
        
        # Update chapters with group_index and video_duration for downstream steps
        for group_idx, group in enumerate(groups):
            for chapter_idx in (group.chapter_indices or []):
                if 0 <= chapter_idx < len(chapters):
                    chapters[chapter_idx]["group_index"] = group_idx
                    # Ensure video_duration is set if not already present
                    if "video_duration" not in chapters[chapter_idx]:
                        chapters[chapter_idx]["video_duration"] = video_duration
        
        # Convert to dictionaries for output
        group_dicts = [group.model_dump() for group in groups]
        
        # Compute metrics
        total_chapters = sum(
            len(g.chapter_indices or []) for g in groups
        )
        avg_per_group = total_chapters / max(len(groups), 1)
        
        logger.info(
            f"Chapter grouping complete: {len(groups)} groups from "
            f"{len(chapters)} chapters (avg {avg_per_group:.1f} chapters/group)"
        )
        
        return StepResult(
            step_id=self.step_id,
            outputs={
                "chapter_groups": group_dicts,
                "group_count": len(groups),
                "groups": groups,  # Keep model instances for downstream steps
                "chapters": chapters,  # Updated chapters with group_index
            },
            metrics={
                "groups_created": len(groups),
                "chapters_processed": len(chapters),
                "avg_chapters_per_group": avg_per_group,
                "similarity_threshold": similarity_threshold,
                "temporal_window": temporal_window,
            },
        )
