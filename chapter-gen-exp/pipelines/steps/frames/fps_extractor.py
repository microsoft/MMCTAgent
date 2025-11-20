"""Concrete FPS-based frame extraction using OpenCV."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from ...utils.chunks import resolve_chunks
from ...utils.transcript_loader import load_transcript


def _infer_duration_seconds(context: StepContext) -> float:
    if context.video_metadata and context.video_metadata.duration_seconds:
        return context.video_metadata.duration_seconds
    if context.video_duration_seconds:
        return context.video_duration_seconds
    transcript = load_transcript(context.transcript_path)
    return transcript.duration


def _clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    for file in path.glob("*"):
        if file.is_file():
            file.unlink()
        else:
            shutil.rmtree(file)


@register_step("frames.fps")
class FPSFrameExtractionStep(PipelineStep):
    """Extracts evenly spaced frames from a local MP4 file."""

    description = "Fixed FPS sampler that writes real frame images to disk."

    def run(self, context: StepContext) -> StepResult:
        fps = float(self.params.get("fps", 1.0))
        max_frames = int(self.params.get("max_frames", 240))
        per_chunk_param = self.params.get("max_frames_per_chunk")
        if per_chunk_param is None:
            max_frames_per_chunk = max_frames
        else:
            max_frames_per_chunk = int(per_chunk_param)
        if max_frames_per_chunk <= 0:
            max_frames_per_chunk = None
        image_format = str(self.params.get("image_format", "jpg")).lower()
        output_subdir = str(self.params.get("output_subdir", "frames"))
        clean_output = bool(self.params.get("clean_output", True))
        chunks_step = self.params.get("chunks_step")

        video_path = Path(context.video_uri).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frames_dir = context.output_dir / output_subdir / self.step_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        if clean_output:
            _clean_output_dir(frames_dir)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file {video_path}")

        native_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (
            frame_count / native_fps
            if native_fps > 0 and frame_count > 0
            else _infer_duration_seconds(context)
        )
        target_interval = 1.0 / max(fps, 0.1)
        native_interval = 1.0 / native_fps if native_fps > 0 else target_interval

        # Resolve chunk metadata so we know which time windows to sample inside.
        chunks = resolve_chunks(context.data_store, chunks_step, duration)
        frames: List[Dict[str, Any]] = []
        chunk_counts: Dict[int, int] = {}
        tolerance = 1e-6
        if not chunks:
            chunks = [{"index": 0, "start": 0.0, "end": duration}]
        chunk_idx = 0
        next_capture_time = chunks[0]["start"]
        frame_index = 0
        captured = 0

        # Walk the video once, advancing chunk pointers as timestamps leave each span.
        while captured < max_frames:
            success, frame = capture.read()
            if not success:
                break

            timestamp = frame_index * native_interval
            while chunk_idx < len(chunks) and timestamp > chunks[chunk_idx]["end"] + tolerance:
                chunk_idx += 1
                if chunk_idx >= len(chunks):
                    break
                next_capture_time = chunks[chunk_idx]["start"]

            if chunk_idx >= len(chunks):
                break

            chunk = chunks[chunk_idx]
            if timestamp + tolerance < chunk["start"]:
                frame_index += 1
                continue

            if timestamp <= chunk["end"] + tolerance and timestamp + 1e-6 >= next_capture_time:
                # Timestamp falls inside the active chunk window, so emit a frame (as long
                # as this chunk has not reached its per-chunk cap) and advance the desired
                # capture time by the configured interval.
                chunk_id = int(chunk["index"])
                current_count = chunk_counts.get(chunk_id, 0)
                if max_frames_per_chunk is not None and current_count >= max_frames_per_chunk:
                    next_capture_time = max(next_capture_time, chunk["end"] + tolerance)
                else:
                    chunk_counts[chunk_id] = current_count + 1
                    file_path = frames_dir / (
                        f"{self.step_id}_chunk{chunk_id:04d}_{chunk_counts[chunk_id]-1:04d}.{image_format}"
                    )
                    saved = cv2.imwrite(str(file_path), frame)
                    if not saved:
                        raise RuntimeError(f"Failed to write frame to {file_path}")
                    frames.append(
                        {
                            "frame_id": f"{self.step_id}-chunk{chunk_id:04d}-{chunk_counts[chunk_id]-1:04d}",
                            "timestamp": round(timestamp, 3),
                            "chunk_index": chunk_id,
                            "path": str(file_path),
                        }
                    )
                    captured += 1
                    next_capture_time += target_interval
                if next_capture_time > chunk["end"] + tolerance and chunk_idx + 1 < len(chunks):
                    next_capture_time = chunks[chunk_idx + 1]["start"]
            frame_index += 1

        capture.release()

        if not frames:
            raise RuntimeError(
                f"No frames extracted from {video_path}. Check FPS parameters or video duration."
            )

        metrics = {
            "frames_produced": float(len(frames)),
            "fps": fps,
            "video_duration": float(duration),
        }
        produced = {
            "frames": frames,
            "frame_interval": target_interval,
            "output_dir": str(frames_dir),
        }
        artifacts = {"frames_dir": str(frames_dir)}

        return StepResult(
            step_id=self.step_id,
            produced=produced,
            metrics=metrics,
            artifacts=artifacts,
        )
