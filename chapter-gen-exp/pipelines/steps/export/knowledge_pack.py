"""Exporter step that writes a consolidated knowledge pack to disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step


@register_step("export.knowledge-pack")
class KnowledgePackExporterStep(PipelineStep):
    """Serializes collected artifacts into a single JSON file."""

    description = "Writes frames, chapters, and metadata into a shareable bundle."

    def run(self, context: StepContext) -> StepResult:
        chapters_step = self.params.get("chapters_step")
        frames_step = self.params.get("frames_step")
        if not chapters_step:
            raise ValueError("'chapters_step' parameter is required")

        chapters_payload = context.data_store.get(chapters_step, {})
        frames_payload = context.data_store.get(frames_step, {}) if frames_step else {}

        knowledge_pack: Dict[str, Any] = {
            "video_uri": context.video_uri,
            "metadata": context.metadata,
            "chapters": chapters_payload.get("chapters", []),
            "frames": frames_payload.get("frames", []),
            "stats": {
                "chapter_count": len(chapters_payload.get("chapters", [])),
                "frame_count": len(frames_payload.get("frames", [])),
            },
        }

        file_name = self.params.get("file_name", f"{self.step_id}.json")
        output_path = context.output_dir / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(knowledge_pack, handle, indent=2)

        return StepResult(
            step_id=self.step_id,
            produced={"knowledge_pack": knowledge_pack},
            metrics={"chapters": float(knowledge_pack["stats"]["chapter_count"]),
                     "frames": float(knowledge_pack["stats"]["frame_count"]),},
            artifacts={"knowledge_pack": str(output_path)},
        )
