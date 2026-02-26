"""Extraction planning pipeline step."""

from typing import Dict, List, Any
from mmct.video_pipeline.core.ingestion.pipelines.steps.base import PipelineStep, StepResult, StepContext
from mmct.video_pipeline.core.ingestion.pipelines.steps.registry import register_step
from mmct.video_pipeline.core.ingestion.models import LLMConstraints, ExtractionPlan
from .planner import AdaptiveExtractionPlanner


@register_step("ingestion.extraction_planning")
class ExtractionPlanningStep(PipelineStep):
    """Compute optimal extraction plan based on LLM constraints and video properties."""
    
    step_type = "ingestion.extraction_planning"
    description = "Plan extraction strategy based on deployment constraints."
    
    def get_param(self, key: str, context: StepContext, default=None):
        """Helper to get parameter from params or user_params."""
        return self.params.get(key, context.user_params.get(key, default))
    
    async def run(self, context: StepContext) -> StepResult:
        # Load constraints from config (with defaults)
        llm_config: Dict[str, Any] = {}
        if hasattr(context.provider, 'llm_config'):
            llm_config = getattr(context.provider, 'llm_config', {}) or {}
        
        constraints: LLMConstraints = LLMConstraints.from_config(llm_config) if llm_config else LLMConstraints()
        
        # Get video properties from previous steps
        video_duration: float = getattr(context, 'video_duration', 0.0)
        
        # Get video_chunking step name from params (allows flexibility)
        video_chunking_step: str = self.get_param("video_chunking_step", context, default="video_chunking")
        video_chunks: List[Dict[str, Any]] = context.data_store.get(video_chunking_step, "video_chunks") or []
        num_chunks: int = len(video_chunks)
        
        transcript_words: int = context.data_store.get("transcribe", "word_count") or 0
        
        # Default frames per chapter - will be refined by dense_keyframes step later
        # This provides an initial estimate for planning
        frames_data: int = self.get_param("default_frames_per_chapter", context, default=8)
        
        # Compute optimal plan
        planner = AdaptiveExtractionPlanner(constraints)
        plan: ExtractionPlan = planner.plan_extraction(
            video_duration_seconds=video_duration,
            num_chapters=num_chunks,
            transcript_word_count=transcript_words,
            available_frames_per_chapter=frames_data,
        )
        
        return StepResult(
            step_id=self.step_id,
            outputs={"extraction_plan": plan},
            metrics={
                "frames_per_chapter": plan.frames_per_chapter,
                "use_unified_extraction": plan.use_unified_extraction,
                "concurrent_requests": plan.concurrent_requests,
            },
        )
