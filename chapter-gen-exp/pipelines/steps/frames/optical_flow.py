"""Optical-flow-based keyframe extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from ...utils.chunks import resolve_chunks


def _calc_scale(width: int, height: int, max_edge: int) -> float:
    longest = max(width, height)
    if longest <= max_edge:
        return 1.0
    return max_edge / float(longest)


def _sample_interval(fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        return 1
    if fps <= 0:
        fps = 30.0
    interval = int(round(fps / float(target_fps)))
    return max(interval, 1)


def _motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


@register_step("frames.optical-flow")
class OpticalFlowFrameExtractionStep(PipelineStep):
    """Extracts keyframes using motion magnitude between sampled frames."""

    description = "Motion-triggered sampler using Farneback optical flow."

    def run(self, context: StepContext) -> StepResult:
        video_path = Path(context.video_uri).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        params = self.params
        threshold = float(params.get("motion_threshold", 0.8))
        sample_fps = float(params.get("sample_fps", 1.0))
        max_edge = int(params.get("max_frame_width", 720))
        max_frames = int(params.get("max_frames", 200))
        per_chunk_param = params.get("max_frames_per_chunk")
        if per_chunk_param is None:
            max_frames_per_chunk = max_frames
        else:
            max_frames_per_chunk = int(per_chunk_param)
        if max_frames_per_chunk <= 0:
            max_frames_per_chunk = None
        output_subdir = str(params.get("output_subdir", "frames"))
        image_format = str(params.get("image_format", "jpg")).lower()
        clean_output = bool(params.get("clean_output", True))
        chunks_step = params.get("chunks_step")

        frames_dir = context.output_dir / output_subdir / self.step_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        if clean_output:
            for file in frames_dir.glob("*"):
                file.unlink()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file {video_path}")

        native_fps = context.video_metadata.fps if context.video_metadata else 0.0
        if not native_fps:
            native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        scale = _calc_scale(width, height, max_edge)
        interval = _sample_interval(native_fps, sample_fps)

        chunks = resolve_chunks(context.data_store, chunks_step, context.video_duration_seconds or 0.0)
        chunk_idx = 0
        chunk_counts: Dict[int, int] = {}
        frames: List[Dict[str, Any]] = []
        prev_gray = None
        frame_idx = -1
        captured = 0
        tolerance = 1e-6

        while captured < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_idx += 1
            timestamp = frame_idx / native_fps if native_fps else 0.0

            while chunk_idx < len(chunks) and timestamp > chunks[chunk_idx]["end"] + tolerance:
                chunk_idx += 1
                prev_gray = None
            if chunk_idx >= len(chunks):
                break

            chunk = chunks[chunk_idx]
            if timestamp + tolerance < chunk["start"]:
                continue

            chunk_start_frame = int(chunk["start"] * native_fps) if native_fps else 0
            frames_since_chunk_start = frame_idx - chunk_start_frame
            if frames_since_chunk_start < 0 or frames_since_chunk_start % interval != 0:
                continue

            if scale < 1.0:
                frame_small = cv2.resize(
                    frame_bgr,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_small = frame_bgr

            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            if prev_gray is None or timestamp - chunk["start"] <= tolerance:
                motion = 0.0
            else:
                motion = _motion_score(prev_gray, gray)

            should_save = prev_gray is None or motion >= threshold
            prev_gray = gray

            if not should_save:
                continue

            chunk_id = int(chunk["index"])
            current_count = chunk_counts.get(chunk_id, 0)
            if max_frames_per_chunk is not None and current_count >= max_frames_per_chunk:
                continue
            chunk_counts[chunk_id] = current_count + 1
            file_path = frames_dir / (
                f"{self.step_id}_chunk{chunk_id:04d}_{chunk_counts[chunk_id]-1:04d}.{image_format}"
            )
            if not cv2.imwrite(str(file_path), frame_bgr):
                raise RuntimeError(f"Failed to write frame to {file_path}")

            frames.append(
                {
                    "frame_id": f"{self.step_id}-chunk{chunk_id:04d}-{chunk_counts[chunk_id]-1:04d}",
                    "timestamp": round(timestamp, 3),
                    "chunk_index": chunk_id,
                    "frame_index": frame_idx,
                    "motion_score": round(motion, 4),
                    "path": str(file_path),
                }
            )
            captured += 1

        capture.release()

        if not frames:
            raise RuntimeError(
                "No keyframes were captured. Try lowering motion_threshold or sample_fps."
            )

        produced = {
            "frames": frames,
            "frame_interval": interval / native_fps if native_fps else None,
            "output_dir": str(frames_dir),
        }
        metrics = {
            "frames_produced": float(len(frames)),
            "motion_threshold": threshold,
            "sample_interval_frames": float(interval),
        }
        artifacts = {"frames_dir": str(frames_dir)}

        return StepResult(
            step_id=self.step_id,
            produced=produced,
            metrics=metrics,
            artifacts=artifacts,
        )
