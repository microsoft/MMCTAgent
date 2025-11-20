"""Transcript preprocessing placeholder implementation."""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step
from ...utils.transcript_loader import TranscriptSegment, load_transcript


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


@register_step("transcript.clean")
class BasicTranscriptCleanerStep(PipelineStep):
    """Lightweight segment cleaner that removes short or empty spans."""

    description = "Cleans transcript segments and emits normalized text."

    def run(self, context: StepContext) -> StepResult:
        transcript = load_transcript(context.transcript_path)
        min_chars = int(self.params.get("min_chars", 5))

        cleaned: List[Dict[str, Any]] = []
        for segment in transcript.segments:
            normalized = _normalize_text(segment.text)
            if len(normalized) < min_chars:
                continue
            cleaned.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": normalized,
                }
            )

        total_duration = transcript.duration
        return StepResult(
            step_id=self.step_id,
            produced={"segments": cleaned, "duration": total_duration},
            metrics={
                "segments_emitted": float(len(cleaned)),
                "duration_seconds": float(total_duration),
            },
        )
