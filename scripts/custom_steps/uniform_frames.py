"""Custom step: Uniform frame extraction + blob upload.

Extracts frames from video at a uniform 1 fps rate using ffmpeg,
then uploads each frame to the configured storage provider.

This is an example of a custom ingestion step that lives outside the
core MMCT library.  Register it by importing this module (or the
``custom_steps`` package) before running the pipeline.

Blob path convention:
    <normalized-video-id>/<timestamp_second>/frame.jpg
"""

import os
import re
import glob as globmod
import asyncio
import subprocess
import shutil
from typing import Dict, List, Any, Optional

from loguru import logger

from mmct.video_pipeline import PipelineStep, StepContext, StepResult, register_step
from mmct.providers.base.storage_provider import BaseStorageProvider


_BLOB_INVALID_RE = re.compile(r'[^a-zA-Z0-9\-_.]')


def normalize_video_id(video_id: str) -> str:
    """Normalize a video ID for use as a blob path segment."""
    return _BLOB_INVALID_RE.sub('_', video_id)


def _extract_frames_at_1fps(
    video_path: str,
    output_dir: str,
    extension: str = "jpg",
) -> List[Dict[str, Any]]:
    """Extract one frame per second from *video_path* using ffmpeg.

    Returns a list of dicts with keys:
        timestamp_second (int), filepath (str), filename (str)
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for uniform frame extraction but was not found on PATH"
        )

    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, f"frame_%06d.{extension}")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", "fps=1",
        "-q:v", "2",
        "-y",
        output_pattern,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr[-500:]}")

    # ffmpeg names frames starting at 1: frame_000001.jpg, frame_000002.jpg, ...
    extracted_files = sorted(
        globmod.glob(os.path.join(output_dir, f"frame_*.{extension}"))
    )

    frames: List[Dict[str, Any]] = []
    for filepath in extracted_files:
        filename = os.path.basename(filepath)
        # frame_000001 → second 0, frame_000002 → second 1, etc.
        seq_num = int(filename.split("_")[1].split(".")[0])
        second = seq_num - 1
        frames.append({
            "timestamp_second": second,
            "filepath": filepath,
            "filename": filename,
        })

    return frames


@register_step("ingestion.uniform_frames")
class UniformFrameExtractionStep(PipelineStep):
    """Extract frames at uniform 1 fps and upload to blob storage.

    Params:
        container_name: Blob container (default: "video-frames-lively")
        extension: Image extension (default: "jpg")
        upload_batch_size: Frames uploaded per batch (default: 20)
        compress_step: Step ID for compressed video (default: "compress")
    """

    step_type = "ingestion.uniform_frames"
    description = "Extract frames at 1 fps and upload to blob storage."

    async def run(self, context: StepContext) -> StepResult:
        container_name: str = self.get_param(
            "container_name", context, default="video-frames-lively"
        )
        extension: str = self.get_param("extension", context, default="jpg")

        compress_step: str = self.get_param("compress_step", context, default="compress")
        effective_video_path: str = (
            context.data_store.get(compress_step, "video_path") or context.video_path
        )

        video_id: str = getattr(context, "video_id", "unknown")
        norm_id = normalize_video_id(video_id)

        # --- 1. Extract frames to local disk ---
        output_base = os.path.join(context.output_dir, "uniform_frames", norm_id)
        # Clean any leftover frames from prior runs
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
        os.makedirs(output_base, exist_ok=True)

        loop = asyncio.get_running_loop()
        frames = await loop.run_in_executor(
            None,
            _extract_frames_at_1fps,
            effective_video_path,
            output_base,
            extension,
        )

        if not frames:
            context.logger.warning("No frames extracted")
            return StepResult(
                step_id=self.step_id,
                outputs={"frames": [], "container_name": container_name},
                metrics={"total_frames": 0, "uploaded": 0},
            )

        context.logger.info(
            f"Extracted {len(frames)} frames at 1 fps from {effective_video_path}"
        )

        # --- 2. Build a storage provider for the dedicated container ---
        storage_provider: Optional[BaseStorageProvider] = getattr(
            context.provider, "storage_provider", None
        )
        if storage_provider is None:
            context.logger.error("No storage provider on context.provider")
            return StepResult(
                step_id=self.step_id,
                outputs={"frames": frames, "error": "No storage provider"},
                metrics={"total_frames": len(frames), "uploaded": 0},
            )

        # --- 3. Upload frames in batches ---
        batch_size: int = self.get_param("upload_batch_size", context, default=20)
        uploaded_count = 0
        failed_count = 0

        async def _upload(frame: Dict[str, Any]) -> None:
            nonlocal uploaded_count, failed_count
            ts = frame["timestamp_second"]
            blob_name = f"{norm_id}/{ts}/frame.{extension}"
            try:
                blob_url = await storage_provider.upload_file(
                    file_name=blob_name,
                    src_file_path=frame["filepath"],
                    folder_name=container_name,
                )
                frame["blob_url"] = blob_url
                frame["blob_name"] = blob_name
                uploaded_count += 1
                if uploaded_count % 500 == 0:
                    logger.info(f"Upload progress: {uploaded_count}/{len(frames)} frames")
            except Exception as exc:
                logger.error(f"Upload failed for ts={ts}: {exc}")
                failed_count += 1

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            await asyncio.gather(*[_upload(f) for f in batch])

        context.logger.info(
            f"Uploaded {uploaded_count}/{len(frames)} frames "
            f"({failed_count} failures) to {container_name}"
        )

        return StepResult(
            step_id=self.step_id,
            outputs={
                "frames": frames,
                "container_name": container_name,
                "normalized_video_id": norm_id,
            },
            metrics={
                "total_frames": len(frames),
                "uploaded": uploaded_count,
                "failed": failed_count,
            },
            artifacts=[f["filepath"] for f in frames],
        )
