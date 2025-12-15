"""Keyframe extraction step wrapper."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from .keyframe_extractor import KeyframeExtractionConfig
from .keyframe_processor import KeyframeProcessor
import asyncio


@register_step("ingestion.keyframes")
class KeyframeExtractionStep(PipelineStep):
    """
    Extract keyframes from video.

    Params:
        source_step: Step ID containing video_path (default: "compress")
        motion_threshold: Motion detection threshold (overridable from user_params)
        sample_fps: Sampling FPS (overridable from user_params)
    """

    step_type = "ingestion.keyframes"
    description = "Extract keyframes from video"

    async def run(self, context: StepContext) -> StepResult:
        """Execute keyframe extraction."""
        # Get source video
        source_step = self.get_param("source_step", context, default="compress")

        # Check for video chunks first
        chunks = context.data_store.get(source_step, "video_chunks")
        video_path = context.data_store.get(source_step, "video_path")

        if not chunks and not video_path:
            video_path = context.video_path

        # Get keyframe config (prefer user_params, then step params, then defaults)
        keyframe_config = context.user_params.get("keyframe_config", {})
        motion_threshold = self.get_param(
            "motion_threshold", context, default=keyframe_config.get("motion_threshold", 1.5)
        )
        sample_fps = self.get_param(
            "sample_fps", context, default=keyframe_config.get("sample_fps", 2)
        )
        max_frame_width = self.get_param(
            "max_frame_width", context, default=keyframe_config.get("max_frame_width", 800)
        )
        num_workers = self.get_param(
            "num_workers", context, default=keyframe_config.get("num_workers", 4)
        )
        device = self.get_param("device", context, default="cpu")

        # Create keyframe config (shared)
        config = KeyframeExtractionConfig(
            motion_threshold=motion_threshold,
            sample_fps=sample_fps,
            max_frame_width=max_frame_width,
            num_workers=num_workers,
            device=device,
        )
        processor = KeyframeProcessor(keyframe_config=config)

        try:
            if chunks:
                context.logger.info(f"Extracting keyframes for {len(chunks)} video chunks...")
                chunk_keyframes = {}
                artifacts = []

                async def process_chunk(chunk):
                    chunk_id = chunk["chunk_id"]
                    path = chunk["path"]
                    # Use unique hash for each chunk to prevent file collision
                    chunk_hash = f"{context.video_id}_{chunk_id}"

                    # Calculate chunk duration if available, else usage None (processor handles it)
                    duration = chunk.get("end_time", 0) - chunk.get("start_time", 0)

                    json_path = await processor.process_keyframes(
                        video_path=path,
                        video_hash_id=chunk_hash,
                        video_duration=duration if duration > 0 else context.video_duration,
                    )
                    return chunk_id, json_path

                # Run parallel
                try:
                    tasks = [process_chunk(chunk) for chunk in chunks]
                    results = await asyncio.gather(*tasks)

                    for chunk_id, json_path in results:
                        chunk_keyframes[chunk_id] = json_path
                        artifacts.append(json_path)

                except Exception as e:
                    # In case of parallel failure, log and re-raise or handle partials?
                    # Let's let it fail for now to be safe.
                    raise e

                return StepResult(
                    step_id=self.step_id,
                    outputs={
                        "chunk_keyframes": chunk_keyframes,
                        "keyframes_extracted": True,
                    },
                    metrics={
                        "motion_threshold": motion_threshold,
                        "processed_chunks": len(chunk_keyframes),
                    },
                    artifacts=artifacts,
                )

            else:
                context.logger.info(f"Extracting keyframes from {video_path} with device={device}")
                # Single video processing
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
