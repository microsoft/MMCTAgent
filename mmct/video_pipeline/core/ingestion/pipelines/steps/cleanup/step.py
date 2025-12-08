"""Cleanup step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .cleanup_manager import CleanupManager


@register_step("ingestion.cleanup")
class CleanupStep(PipelineStep):
    """
    Clean up temporary files generated during processing.

    Params:
        keep_keyframes: Whether to keep keyframe images (default: False)
    """

    step_type = "ingestion.cleanup"
    description = "Clean up temporary files"

    async def run(self, context: StepContext) -> StepResult:
        """Execute cleanup."""
        keep_keyframes = self.get_param("keep_keyframes", context, default=False)

        context.logger.info(f"Starting cleanup for video: {context.video_id}")

        try:
            # Create cleanup manager
            cleanup_manager = CleanupManager(keep_keyframes=keep_keyframes)

            # Clean up
            await cleanup_manager.cleanup(context.video_id)

            context.logger.info("Cleanup completed successfully")

            return StepResult(
                step_id=self.step_id,
                outputs={"cleanup_completed": True},
                metrics={},
                artifacts=[],
            )

        except Exception as e:
            context.logger.exception(f"Cleanup step failed: {e}")
            raise
