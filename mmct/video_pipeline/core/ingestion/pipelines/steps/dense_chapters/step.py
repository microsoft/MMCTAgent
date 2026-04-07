"""Dense chapter generation with multimodal extraction and parallel processing."""

from typing import Optional, List, Dict, Any, Tuple
from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.video_pipeline.core.ingestion.models import (
    ExtractionCircuitBreaker, 
    ExtractionPlan,
    DenseChapterResponse,
)
from mmct.providers.base import BaseLLMProvider
from .unified_extractor import (
    extract_chapters_parallel, 
    DEFAULT_PARALLEL_CHUNKS,
    DEFAULT_MAX_FRAMES_PER_CHAPTER,
)


@register_step("ingestion.dense_chapters")
class DenseChapterGenerationStep(PipelineStep):
    """
    Dense chapter generation with multimodal extraction.
    
    Features:
    - Multimodal LLM (frames + transcript) for accurate extraction
    - Frames stacked chronologically for visual continuity
    - Parallel processing of chunks
    - Returns DenseChapterResponse instances
    
    Params:
        source_chunks_step: Step ID for video chunks (default: "video_chunking")
        source_keyframes_step: Step ID for keyframes (default: "dense_keyframes")
        parallel_chunks: Parallel batch size (default: 4)
        max_frames_per_chapter: Max frames per chapter (default: 12)
    """
    
    step_type = "ingestion.dense_chapters"
    description = "Generate dense chapters with multimodal extraction."
    
    async def run(self, context: StepContext) -> StepResult:
        """Execute dense chapter generation."""
        # Get extraction plan
        extraction_plan: Optional[ExtractionPlan] = context.data_store.get(
            "extraction_planning", "extraction_plan"
        )
        
        # Initialize circuit breaker
        circuit_breaker = ExtractionCircuitBreaker(
            failure_threshold=extraction_plan.circuit_breaker_failure_threshold if extraction_plan else 5,
            reset_seconds=extraction_plan.circuit_breaker_reset_seconds if extraction_plan else 60,
        )
        
        # Get inputs
        source_chunks_step = self.get_param("source_chunks_step", context, default="video_chunking")
        video_chunks: List[Dict[str, Any]] = context.data_store.get(source_chunks_step, "video_chunks") or []
        
        source_keyframes_step = self.get_param("source_keyframes_step", context, default="dense_keyframes")
        keyframes_data: List[Dict[str, Any]] = context.data_store.get(source_keyframes_step, "keyframes_per_chunk")
        if not keyframes_data:
            keyframes_data = context.data_store.get("keyframes", "keyframes_per_chunk") or []
        
        if not video_chunks:
            context.logger.warning("No video chunks found")
            return StepResult(
                step_id=self.step_id,
                outputs={"chapters": [], "failed_chapters": []},
                metrics={"total_chapters": 0, "success_rate": 0.0},
                success=False,
            )
        
        # Get config
        llm_provider: BaseLLMProvider = context.provider.llm_provider
        parallel_chunks = self.get_param("parallel_chunks", context, default=DEFAULT_PARALLEL_CHUNKS)
        max_frames = self.get_param("max_frames_per_chapter", context, default=DEFAULT_MAX_FRAMES_PER_CHAPTER)
        
        context.logger.info(f"Processing {len(video_chunks)} chunks (parallel={parallel_chunks}, max_frames={max_frames})")
        
        # Prepare chunks with keyframes
        chunks_with_keyframes: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = [
            (idx, chunk, keyframes_data[idx] if idx < len(keyframes_data) else {})
            for idx, chunk in enumerate(video_chunks)
        ]
        
        # Extract chapters (returns dicts with chunk metadata merged)
        chapters, failed = await extract_chapters_parallel(
            chunks_with_keyframes=chunks_with_keyframes,
            llm_provider=llm_provider,
            circuit_breaker=circuit_breaker,
            extraction_plan=extraction_plan,
            parallel_chunks=parallel_chunks,
            max_frames_per_chapter=max_frames,
        )
        
        # Add video_duration to each chapter for temporal query support
        video_duration = context.video_duration
        for chapter in chapters:
            chapter["video_duration"] = video_duration
        
        # Calculate metrics
        success_rate = len(chapters) / max(len(video_chunks), 1)
        threshold = extraction_plan.partial_success_threshold if extraction_plan else 0.7
        
        context.logger.info(f"Done: {len(chapters)}/{len(video_chunks)} succeeded ({success_rate:.1%})")
        
        return StepResult(
            step_id=self.step_id,
            outputs={
                "chapters": chapters,
                "raw_chapters": chapters,  # Alias for compatibility
                "failed_chapters": failed,
            },
            metrics={
                "total_chapters": len(chapters),
                "failed_chapters": len(failed),
                "success_rate": success_rate,
            },
        )
