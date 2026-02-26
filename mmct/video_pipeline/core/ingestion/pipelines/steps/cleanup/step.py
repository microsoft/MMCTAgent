"""Cleanup step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .cleanup_manager import CleanupManager


@register_step("ingestion.cleanup")
class CleanupStep(PipelineStep):
    """
    Clean up temporary files generated during processing.

    Cleans up all temporary files created during the ingestion pipeline:
    - Compressed video files (media/compressed/{video_id}.mp4)
    - Audio files extracted for transcription (media/{video_id}.wav, .mp3)
    - Dense keyframe images (media/dense_keyframes/{chunk_id}/*.jpg)
    - Transcript files (media/transcript_{video_id}.srt)
    - Legacy chapter/object JSON files

    Params:
        keep_keyframes: Whether to keep keyframe images (default: False)
        keep_compressed: Whether to keep compressed video (default: False)
    """

    step_type = "ingestion.cleanup"
    description = "Clean up temporary files"

    async def run(self, context: StepContext) -> StepResult:
        """Execute cleanup."""
        keep_keyframes = self.get_param("keep_keyframes", context, default=False)
        keep_compressed = self.get_param("keep_compressed", context, default=False)

        context.logger.debug(f"Starting cleanup for video: {context.video_id}")

        try:
            # Create cleanup manager with output_dir for dense_keyframes cleanup
            cleanup_manager = CleanupManager(
                keep_keyframes=keep_keyframes,
                keep_compressed=keep_compressed,
                output_dir=context.output_dir,
            )

            # Clean up
            deleted_count = await cleanup_manager.cleanup(context.video_id)

            context.logger.debug(f"Cleanup completed: {deleted_count} items removed")

            return StepResult(
                step_id=self.step_id,
                outputs={"cleanup_completed": True, "items_deleted": deleted_count},
                metrics={"items_deleted": deleted_count},
                artifacts=[],
            )

        except Exception as e:
            context.logger.exception(f"Cleanup step failed: {e}")
            raise
