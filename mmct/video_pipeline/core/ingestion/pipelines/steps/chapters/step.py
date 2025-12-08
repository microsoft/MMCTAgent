"""Chapter generation step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .chapter_ingestion_pipeline import ChapterIngestionPipeline


@register_step("ingestion.chapters")
class ChapterGenerationStep(PipelineStep):
    """
    Generate semantic chapters from transcript and keyframes.

    Params:
        transcript_step: Step ID containing transcript (default: "transcribe")
        keyframes_step: Step ID containing keyframes (default: "keyframes")
    """

    step_type = "ingestion.chapters"
    description = "Generate semantic chapters from transcript"

    async def run(self, context: StepContext) -> StepResult:
        """Execute chapter generation."""
        # Get transcript from previous step or context
        transcript_step = self.get_param("transcript_step", context, default="transcribe")
        transcript = context.data_store.get(transcript_step, "transcript")

        if not transcript:
            raise ValueError(f"Transcript not found from step: {transcript_step}")

        # Get keyframe blob URL
        blob_manager = context.provider.storage_provider
        keyframe_blob_url = await blob_manager.get_file_url(file_name=f"{context.video_id}")

        # Get frame stacking config
        frame_stacking_grid_size = context.user_params.get("frame_stacking_grid_size", 4)

        context.logger.info(f"Generating chapters for video: {context.video_id}")

        try:
            # Create chapter pipeline
            chapter_pipeline = ChapterIngestionPipeline(
                hash_id=context.video_id,
                keyframe_index_name=context.provider.vectordb_keyframes.index_name,
                transcript=transcript,
                keyframe_blob_url=keyframe_blob_url,
                frame_stacking_grid_size=frame_stacking_grid_size,
                video_duration=context.video_duration,
                llm_provider=context.provider.llm_provider,
                embedding_provider=context.provider.embedding_provider,
            )

            # Run chapter generation
            chapter_responses, chapter_transcripts, is_already_ingested = await chapter_pipeline.run(
                url=context.url
            )

            context.logger.info(f"Generated {len(chapter_responses)} chapters")

            return StepResult(
                step_id=self.step_id,
                outputs={
                    "chapter_responses": chapter_responses,
                    "chapter_transcripts": chapter_transcripts,
                    "is_already_ingested": is_already_ingested,
                    "chapters_generated": True,
                },
                metrics={"num_chapters": len(chapter_responses)},
                artifacts=[],
            )

        except Exception as e:
            context.logger.exception(f"Chapter generation failed: {e}")
            raise
