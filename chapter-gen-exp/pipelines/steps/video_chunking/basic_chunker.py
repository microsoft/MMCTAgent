"""Basic video chunking step."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step


@dataclass
class VideoChunk:
    index: int
    start: float
    end: float


@register_step("video.chunk.basic")
class BasicVideoChunkStep(PipelineStep):
    """Produces a single chunk spanning the entire video."""

    description = "Placeholder chunker that treats the full video as one chunk."

    def run(self, context: StepContext) -> StepResult:
        duration = context.video_duration_seconds or 0.0
        chunk = VideoChunk(index=0, start=0.0, end=duration)
        chunks = [chunk.__dict__]
        return StepResult(
            step_id=self.step_id,
            produced={"chunks": chunks},
            metrics={"chunk_count": 1.0, "duration_seconds": duration},
        )
