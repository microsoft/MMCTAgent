"""Keyframe extraction step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .keyframe_extractor import KeyframeExtractionConfig
from .keyframe_processor import KeyframeProcessor


@register_step("ingestion.keyframes")
class KeyframeExtractionStep(PipelineStep):
    """
    Extract keyframes from video with optional time offset.

    Params:
        source_step: Step ID containing video_path (default: "compress")
        apply_time_offset: Whether to apply time offset for video parts (default: True)
        motion_threshold: Motion detection threshold (overridable from user_params)
        sample_fps: Sampling FPS (overridable from user_params)
    """

    step_type = "ingestion.keyframes"
    description = "Extract keyframes from video"

    async def run(self, context: StepContext) -> StepResult:
        """Execute keyframe extraction."""
        # Get source video
        source_step = self.get_param("source_step", context, default="compress")
        video_path = context.data_store.get(source_step, "video_path")

        if not video_path:
            video_path = context.video_path

        # Get keyframe config (prefer user_params, then step params, then defaults)
        keyframe_config = context.user_params.get("keyframe_config", {})
        motion_threshold = self.get_param(
            "motion_threshold", context, default=keyframe_config.get("motion_threshold", 1.5)
        )
        sample_fps = self.get_param(
            "sample_fps", context, default=keyframe_config.get("sample_fps", 2)
        )

        context.logger.info(
            f"Extracting keyframes from {video_path} with motion_threshold={motion_threshold}, "
            f"sample_fps={sample_fps}"
        )

        try:
            # Create keyframe config
            config = KeyframeExtractionConfig(
                motion_threshold=motion_threshold,
                sample_fps=sample_fps,
            )

            # Create processor
            processor = KeyframeProcessor(keyframe_config=config)

            # Process keyframes
            keyframe_json_path = await processor.process_keyframes(
                video_path=video_path,
                video_hash_id=context.video_id,
                video_duration=context.video_duration,
            )

            context.logger.info(f"Keyframes extracted and saved to: {keyframe_json_path}")

            return StepResult(
                step_id=self.step_id,
                outputs={
                    "keyframe_json_path": keyframe_json_path,
                    "keyframes_extracted": True,
                },
                metrics={
                    "motion_threshold": motion_threshold,
                    "sample_fps": sample_fps,
                },
                artifacts=[keyframe_json_path],
            )

        except Exception as e:
            context.logger.exception(f"Keyframe extraction failed: {e}")
            raise
