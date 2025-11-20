"""Sequential chapter generator placeholder."""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step


def _summarize_text(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


@register_step("chapters.sequential")
class SequentialChapterGenerationStep(PipelineStep):
    """Groups transcript segments into chapters using greedy windows."""

    description = "Greedy sequential chunker intended to be swapped with LLM-backed versions."

    def run(self, context: StepContext) -> StepResult:
        source_step = self.params.get("source_step")
        if not source_step:
            raise ValueError("'source_step' param is required for chapter generation")

        source_payload = context.data_store.get(source_step, {})
        segments: List[Dict[str, Any]] = source_payload.get("segments", [])
        if not segments:
            raise ValueError(
                f"Step '{self.step_id}' expected transcript segments from '{source_step}'"
            )

        max_duration = float(self.params.get("max_duration", 60.0))
        max_segments = int(self.params.get("max_segments", 5))

        chapters: List[Dict[str, Any]] = []
        bucket: List[Dict[str, Any]] = []
        bucket_start = None

        for segment in segments:
            if not bucket:
                bucket_start = segment["start"]
            bucket.append(segment)
            current_duration = segment["end"] - (bucket_start or segment["start"])
            if len(bucket) >= max_segments or current_duration >= max_duration:
                chapters.append(self._build_chapter(bucket))
                bucket = []
                bucket_start = None

        if bucket:
            chapters.append(self._build_chapter(bucket))

        return StepResult(
            step_id=self.step_id,
            produced={"chapters": chapters},
            metrics={"chapters_emitted": float(len(chapters))},
        )

    def _build_chapter(self, bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
        start = bucket[0]["start"]
        end = bucket[-1]["end"]
        combined_text = " ".join(item["text"] for item in bucket)
        summary = _summarize_text(combined_text)
        return {
            "title": summary.split(".")[0][:60] or "Auto Chapter",
            "start": start,
            "end": end,
            "summary": summary,
            "segment_count": len(bucket),
        }
