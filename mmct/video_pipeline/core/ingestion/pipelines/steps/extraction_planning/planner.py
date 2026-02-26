"""Adaptive extraction planner that computes optimal strategy based on constraints."""

from mmct.video_pipeline.core.ingestion.models import LLMConstraints, ExtractionPlan


class AdaptiveExtractionPlanner:
    """
    Plans extraction strategy based on LLM deployment constraints.
    
    Considers:
    - Token limits (context and output)
    - Rate limits (RPM, TPM)
    - Video properties (duration, chapters, transcript)
    - Available frames per chapter
    """
    
    def __init__(self, constraints: LLMConstraints):
        self.constraints = constraints
    
    def plan_extraction(
        self,
        video_duration_seconds: float,
        num_chapters: int,
        transcript_word_count: int,
        available_frames_per_chapter: int,
    ) -> ExtractionPlan:
        """
        Compute optimal extraction plan.
        
        Returns ExtractionPlan with:
        - frames_per_chapter: Optimal frames to include
        - use_unified_extraction: Single vs batched extraction
        - concurrent_requests: Parallelism level
        - batch_size: Chapters per batch
        - Error recovery parameters
        """
        # Estimate tokens per frame (~100 tokens for base64 image)
        tokens_per_frame = 100
        
        # Estimate tokens for transcript (1.3 tokens per word)
        transcript_tokens = int(transcript_word_count * 1.3)
        transcript_tokens_per_chapter = transcript_tokens // max(num_chapters, 1)
        
        # Calculate max frames that fit in context
        available_context = self.constraints.max_context_tokens - transcript_tokens_per_chapter - 2000  # Reserve for prompt
        max_frames = min(
            available_context // tokens_per_frame,
            available_frames_per_chapter,
        )
        
        # Determine if unified extraction is feasible
        # Unified needs ~3000 output tokens, batched needs ~1000 per batch
        use_unified = self.constraints.max_output_tokens >= 3000
        
        # Calculate concurrent requests based on rate limits
        # Conservative: use 50% of RPM limit
        concurrent_requests = max(1, self.constraints.rate_limit_rpm // 2 // 60)
        
        # Batch size for rate limiting
        batch_size = max(1, num_chapters // 4)
        
        return ExtractionPlan(
            frames_per_chapter=max(4, min(max_frames, 16)),
            use_unified_extraction=use_unified,
            concurrent_requests=min(concurrent_requests, 4),
            batch_size=batch_size,
            estimated_tokens_per_chapter=max_frames * tokens_per_frame + transcript_tokens_per_chapter,
            # Error recovery defaults
            fallback_to_basic_on_failure=True,
            max_retries_per_chapter=3,
            partial_success_threshold=0.7,
            circuit_breaker_failure_threshold=5,
            circuit_breaker_reset_seconds=60,
        )
