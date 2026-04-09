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
    Compress video if file size exceeds threshold or codec is incompatible.

    Videos encoded with codecs that OpenCV cannot decode (e.g. AV1) are
    always transcoded to H.264 regardless of file size so that downstream
    keyframe-extraction steps can read them.

    Params:
        max_size_mb: Maximum video size in MB before compression (default: 500)
        target_size_mb: Target size after compression (default: 500)
    """

    step_type = "ingestion.compress"
    description = "Compress video if size exceeds threshold or codec is incompatible"

    async def run(self, context: StepContext) -> StepResult:
        """Execute video compression if needed."""
        max_size_mb = self.get_param("max_size_mb", context, default=500)
        target_size_mb = self.get_param("target_size_mb", context, default=500)
        device = self.get_param("device", context, default="auto")

        # Single video compression
        video_path = context.video_path
        context.logger.info(f"Compressing single video: {os.path.basename(video_path)}")

        compressed_path = await self._compress_single(
            video_path, max_size_mb, target_size_mb, device, context
        )

        return StepResult(
            step_id=self.step_id,
            outputs={"video_path": compressed_path},
            metrics={},
            artifacts=[compressed_path] if compressed_path != video_path else [],
        )

    async def _compress_single(
        self,
        video_path: str,
        max_size_mb: float,
        target_size_mb: float,
        device: str,
        context: StepContext,
    ) -> str:
        try:
            if not os.path.exists(video_path):
                context.logger.warning(f"Video file not found: {video_path}")
                return video_path

            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            # Build a lightweight compressor to probe the input codec.
            media_folder = await get_media_folder()
            compressed_dir = os.path.join(media_folder, "compressed")
            os.makedirs(compressed_dir, exist_ok=True)

            compressor = VideoCompressor(
                input_path=video_path,
                target_size_mb=target_size_mb,
                output_dir=compressed_dir,
                device=device,
            )

            needs_transcode = compressor.needs_transcode()
            needs_compress = file_size_mb > max_size_mb

            if needs_transcode:
                codec = compressor.get_video_codec()
                context.logger.info(
                    f"Codec '{codec}' is not supported by OpenCV — transcoding to H.264"
                )

            if not needs_transcode and not needs_compress:
                return video_path

            await asyncio.to_thread(compressor.compress)

            if os.path.exists(compressor.output_path):
                context.logger.info(
                    f"Compressed {os.path.basename(video_path)}: {file_size_mb:.2f}MB -> {os.path.getsize(compressor.output_path)/(1024*1024):.2f}MB"
                )
                return compressor.output_path

            return video_path
        except Exception as e:
            context.logger.error(f"Compression failed for {video_path}: {e}")
            return video_path
