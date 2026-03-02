"""Uniform frame extraction step.

Extracts frames from video at a uniform 1 fps rate (no filtering),
then uploads each frame to Azure Blob Storage container 'video-frames-lively'.

Blob path convention:
    <normalized-video-id>/<timestamp_second>/frame.jpg

The step creates its own AzureStorageProvider instance targeting the
dedicated container (created automatically if it does not exist).
"""

import os
import re
import cv2
import asyncio
from typing import Dict, List, Any, Optional

from loguru import logger

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from mmct.providers.azure_providers.storage_provider import AzureStorageProvider


# Characters invalid in Azure blob path segments (control chars, backslash, etc.)
# Hyphens, underscores, dots, and alphanumerics are kept as-is.
_BLOB_INVALID_RE = re.compile(r'[^a-zA-Z0-9\-_.]')


def normalize_video_id(video_id: str) -> str:
    """Normalize a video ID for use as a blob path segment.

    Keeps hyphens, underscores, dots, and alphanumerics.
    Replaces any other character with an underscore.
    """
    return _BLOB_INVALID_RE.sub('_', video_id)

def _extract_frames_at_1fps(
    video_path: str,
    output_dir: str,
    extension: str = "jpg",
) -> List[Dict[str, Any]]:
    """Extract one frame per second from *video_path*.

    Returns a list of dicts with keys:
        timestamp_second (int), filepath (str), filename (str)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    os.makedirs(output_dir, exist_ok=True)

    frames: List[Dict[str, Any]] = []
    second = 0

    while second <= int(duration):
        frame_idx = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            break

        filename = f"frame_{second:06d}.{extension}"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame_bgr)

        frames.append({
            "timestamp_second": second,
            "filepath": filepath,
            "filename": filename,
        })
        second += 1

    cap.release()
    return frames


@register_step("ingestion.uniform_frames")
class UniformFrameExtractionStep(PipelineStep):
    """Extract frames at uniform 1 fps and upload to blob storage.

    Params:
        container_name: Blob container (default: "video-frames-lively")
        extension: Image extension (default: "jpg")
        upload_batch_size: Frames uploaded per batch (default: 5)
        compress_step: Step ID for compressed video (default: "compress")
    """

    step_type = "ingestion.uniform_frames"
    description = "Extract frames at 1 fps and upload to blob storage."

    async def run(self, context: StepContext) -> StepResult:
        container_name: str = self.get_param(
            "container_name", context, default="video-frames-lively"
        )
        extension: str = self.get_param("extension", context, default="jpg")

        # Use compressed video when available
        compress_step: str = self.get_param("compress_step", context, default="compress")
        effective_video_path: str = (
            context.data_store.get(compress_step, "video_path") or context.video_path
        )

        video_id: str = getattr(context, "video_id", "unknown")
        norm_id = normalize_video_id(video_id)

        # --- 1. Extract frames to local disk ---
        output_base = os.path.join(context.output_dir, "uniform_frames")
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
        storage_provider: Optional[AzureStorageProvider] = getattr(
            context.provider, "storage_provider", None
        )
        if storage_provider is None:
            context.logger.error("No storage provider on context.provider")
            return StepResult(
                step_id=self.step_id,
                outputs={"frames": frames, "error": "No storage provider"},
                metrics={"total_frames": len(frames), "uploaded": 0},
            )

        # Create a new provider instance reusing the same account / credentials
        # but targeting the dedicated container.
        frame_storage = AzureStorageProvider(
            storage_account_name=storage_provider.storage_account_name,
            keyframe_container_name=container_name,
            credentials=storage_provider.credentials,
            blob_connection_string=storage_provider.blob_connection_string
            if not storage_provider.credentials else None,
        )

        # --- 3. Upload frames in batches ---
        batch_size: int = self.get_param("upload_batch_size", context, default=5)
        uploaded_count = 0
        failed_count = 0

        async def _upload(frame: Dict[str, Any]) -> None:
            nonlocal uploaded_count, failed_count
            ts = frame["timestamp_second"]
            blob_name = f"{norm_id}/{ts}/frame.{extension}"
            try:
                blob_url = await frame_storage.upload_file(
                    file_name=blob_name,
                    src_file_path=frame["filepath"],
                    folder_name=container_name,
                )
                frame["blob_url"] = blob_url
                frame["blob_name"] = blob_name
                uploaded_count += 1
            except Exception as exc:
                logger.error(f"Upload failed for ts={ts}: {exc}")
                failed_count += 1

        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            await asyncio.gather(*[_upload(f) for f in batch])

        await frame_storage.close()

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
