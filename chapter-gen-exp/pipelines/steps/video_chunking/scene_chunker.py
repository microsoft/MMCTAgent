"""Scene-detection-driven video chunking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from scenedetect import SceneManager, VideoManager
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step


@dataclass
class SceneChunk:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _build_detector(params: Dict[str, Any], frame_rate: float):
    detector_type = str(params.get("detector", "content")).lower()
    min_scene_len_seconds = float(params.get("min_scene_length", 2.0))
    min_scene_len_frames = max(1, int(frame_rate * min_scene_len_seconds))

    if detector_type == "adaptive":
        threshold = float(params.get("threshold", 3.0))
        return AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len_frames,
        )

    threshold = float(params.get("threshold", 30.0))
    return ContentDetector(
        threshold=threshold,
        min_scene_len=min_scene_len_frames,
    )


def _detect_scenes(video_path: Path, params: Dict[str, Any]) -> List[SceneChunk]:
    video_manager = VideoManager([str(video_path)])
    downscale = int(params.get("downscale", 2))
    if downscale > 1:
        video_manager.set_downscale_factor(downscale)

    scene_manager = SceneManager()
    detector = _build_detector(params, video_manager.get_base_timecode().framerate)
    scene_manager.add_detector(detector)

    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        base_timecode = video_manager.get_base_timecode()
        detected = scene_manager.get_scene_list(base_timecode)
    finally:
        video_manager.release()

    chunks: List[SceneChunk] = []
    for idx, (start_tc, end_tc) in enumerate(detected):
        chunks.append(
            SceneChunk(
                index=idx,
                start=start_tc.get_seconds(),
                end=end_tc.get_seconds(),
            )
        )
    return chunks


@register_step("video.chunk.scene")
class SceneChunkingStep(PipelineStep):
    """Generates chunk boundaries using PySceneDetect."""

    description = "Scene-aware chunking that emits per-scene spans for downstream steps."

    def run(self, context: StepContext) -> StepResult:
        video_path = Path(context.video_uri).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        params = self.params
        max_scenes = int(params.get("max_scenes", 1000))

        chunks = _detect_scenes(video_path, params)
        if not chunks:
            # fallback to single chunk covering entire video duration
            duration = context.video_duration_seconds or 0.0
            chunks = [SceneChunk(index=0, start=0.0, end=duration)]

        chunks = chunks[:max_scenes]
        serialized = [
            {
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
                "duration": chunk.duration,
            }
            for chunk in chunks
        ]

        for chunk in chunks:
            print(
                f"[video.chunk.scene] Chunk {chunk.index:04d}: "
                f"{chunk.start:.2f}s -> {chunk.end:.2f}s (duration {chunk.duration:.2f}s)"
            )

        metrics = {
            "chunk_count": float(len(serialized)),
            "duration_seconds": float(serialized[-1]["end"]) if serialized else 0.0,
        }

        return StepResult(
            step_id=self.step_id,
            produced={"chunks": serialized},
            metrics=metrics,
        )
