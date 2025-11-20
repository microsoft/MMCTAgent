"""Video metadata helpers for experimentation pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMetadata:
    duration_seconds: float
    fps: float
    frame_count: int


def extract_video_metadata(path: str | Path) -> VideoMetadata:
    video_path = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = 0.0
    if fps > 0.0 and frame_count > 0:
        duration = frame_count / fps
    else:
        duration = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

    capture.release()

    if duration <= 0.0:
        raise RuntimeError(
            f"Could not determine duration for video {video_path}. fps={fps}, frames={frame_count}"
        )

    return VideoMetadata(duration_seconds=duration, fps=fps, frame_count=frame_count)
