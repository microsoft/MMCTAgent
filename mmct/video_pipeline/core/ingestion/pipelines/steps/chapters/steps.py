"""Chapter generation step."""

from typing import Any, Dict, List, Optional
from loguru import logger
from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .llm_scene import SceneLLMChapterGenerator
from .timeline_summary import ChapterTimelineSummarizer


@register_step("ingestion.chapters")
class ChapterGenerationStep(PipelineStep):
    """
    Generate semantic chapters from transcript and keyframes using Scene LLM,
    and then generate a timeline summary.
    """

    step_type = "ingestion.chapters"
    description = "Generate semantic chapters and timeline summary."

    async def run(self, context: StepContext) -> StepResult:
        video_chunking_step = self.get_param(
            "video_chunking_step", context, default="video_chunking"
        )
        video_chunks = context.data_store.get(video_chunking_step, "video_chunks")

        if not video_chunks:
            logger.warning("No video chunks found.")
            return StepResult(
                step_id=self.step_id, outputs={"raw_chapters": [], "video_summary": ""}, metrics={}
            )

        max_concurrent_requests = self.get_param("max_concurrent_requests", context, default=3)

        collect_object_collection_param = self.get_param(
            "collect_object_collection", context, default=True
        )
        # Handle both boolean and dict (new config structure)
        if isinstance(collect_object_collection_param, dict):
            collect_object_collection = (
                str(collect_object_collection_param.get("enabled", "true")).lower() == "true"
            )
        else:
            collect_object_collection = bool(collect_object_collection_param)

        logger.info(f"Generating chapters for {len(video_chunks)} chunks using Scene LLM.")

        # Prepare chunks for SceneLLM
        # SceneLLM expects list of dicts with 'index', 'start', 'end', 'transcript' (dict with 'text'), 'sentence'
        formatted_chunks = []
        for i, chunk_data in enumerate(video_chunks):
            transcript_text = chunk_data.get("transcript") or chunk_data.get("sentence") or ""
            formatted_chunks.append(
                {
                    "index": i,
                    "start": chunk_data["start_time"],
                    "end": chunk_data["end_time"],
                    "transcript": {"text": transcript_text},
                    "sentence": transcript_text,
                }
            )

        # --- 1. Scene LLM Chapter Generation ---
        scene_llm = SceneLLMChapterGenerator(step_id=self.step_id)

        step_params = {
            "max_parallel_requests": max_concurrent_requests,
            **self.params,
            "collect_object_collection": collect_object_collection,
        }

        # SceneLLM uses video_id to find keyframes via get_media_folder if frames not passed explicitly
        raw_chapters = await scene_llm.run_direct(
            chunks=formatted_chunks,
            llm_provider=context.provider.llm_provider,
            video_id=context.video_id,
            params=step_params,
        )

        logger.info(f"Generated {len(raw_chapters)} raw chapters.")

        if collect_object_collection:
            # --- 2. Timeline Summary Generation ---
            logger.debug("Generating timeline summary.")
            timeline_summary_step = ChapterTimelineSummarizer()

            timeline_summary_result = await timeline_summary_step.run_direct(
                chapters=raw_chapters,
                llm_provider=context.provider.llm_provider,
                params=step_params,
            )

            aggregated_video_summary = timeline_summary_result.get("global_summary", "")
            logger.debug(
                f"Generated video summary (length: {len(aggregated_video_summary)} chars)."
            )
        else:
            aggregated_video_summary = ""
            logger.debug("Skipping timeline summary generation (collect_object_collection=False).")

        return StepResult(
            step_id=self.step_id,
            outputs={
                "raw_chapters": raw_chapters,
                "video_summary": aggregated_video_summary,
            },
            metrics={
                "num_chapters": len(raw_chapters),
                "summary_length": len(aggregated_video_summary),
            },
        )
