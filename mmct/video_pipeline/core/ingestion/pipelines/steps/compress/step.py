"""Compression step wrapper."""

import os
import asyncio
from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.video_pipeline.utils.helper import get_media_folder
from .video_compression import VideoCompressor


@register_step("ingestion.compress")
class CompressionStep(PipelineStep):
    """
    Compress video if file size exceeds threshold.

    Params:
        max_size_mb: Maximum video size in MB before compression (default: 500)
        target_size_mb: Target size after compression (default: 500)
    """

    step_type = "ingestion.compress"
    description = "Compress video if size exceeds threshold"

    async def run(self, context: StepContext) -> StepResult:
        """Execute video compression if needed."""
        max_size_mb = self.get_param("max_size_mb", context, default=500)
        target_size_mb = self.get_param("target_size_mb", context, default=500)
        device = self.get_param("device", context, default="auto")

        video_path = context.video_path
        context.logger.info(
            f"Checking video compression for: {os.path.basename(video_path)} (device={device})"
        )

        try:
            # Check file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            context.logger.info(f"Video file size: {file_size_mb:.2f} MB")

            compressed_path = video_path
            compression_ratio = 1.0

            if file_size_mb > max_size_mb:
                context.logger.info(
                    f"Video size ({file_size_mb:.2f} MB) exceeds {max_size_mb} MB. Compressing..."
                )

                # Create compressed directory
                media_folder = await get_media_folder()
                compressed_dir = os.path.join(media_folder, "compressed")
                os.makedirs(compressed_dir, exist_ok=True)

                # Initialize compressor
                compressor = VideoCompressor(
                    input_path=video_path,
                    target_size_mb=target_size_mb,
                    output_dir=compressed_dir,
                    device=device,
                )

                # Compress in thread pool
                await asyncio.to_thread(compressor.compress)

                # Check if compression succeeded
                if os.path.exists(compressor.output_path):
                    compressed_size_mb = os.path.getsize(compressor.output_path) / (1024 * 1024)
                    compression_ratio = compressed_size_mb / file_size_mb
                    compressed_path = compressor.output_path

                    context.logger.info(
                        f"Compression successful: {file_size_mb:.2f} MB → {compressed_size_mb:.2f} MB "
                        f"(ratio: {compression_ratio:.2%})"
                    )
                else:
                    context.logger.warning("Compression failed, using original video")
            else:
                context.logger.info("Video size within limits, no compression needed")

            return StepResult(
                step_id=self.step_id,
                outputs={"video_path": compressed_path},
                metrics={
                    "original_size_mb": file_size_mb,
                    "compressed_size_mb": os.path.getsize(compressed_path) / (1024 * 1024),
                    "compression_ratio": compression_ratio,
                },
                artifacts=[compressed_path] if compressed_path != video_path else [],
            )

        except Exception as e:
            context.logger.exception(f"Compression step failed: {e}")
            # On failure, return original video path
            return StepResult(
                step_id=self.step_id,
                outputs={"video_path": video_path},
                metrics={},
                artifacts=[],
            )
