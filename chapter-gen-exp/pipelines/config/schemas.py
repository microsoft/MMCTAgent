"""Configuration schemas and loader helpers for experimentation pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class StepConfig:
    """Configuration payload for a single pipeline step."""

    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Ordered collection of steps plus execution hints."""

    name: str = "default-pipeline"
    mode: str = "sequential"
    steps: List[StepConfig] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Top-level configuration for a video processing experiment."""

    video_uri: str
    transcript_path: str
    output_dir: str
    video_duration_seconds: Optional[float] = None
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _parse_step_configs(raw_steps: List[Dict[str, Any]]) -> List[StepConfig]:
    parsed: List[StepConfig] = []
    for idx, data in enumerate(raw_steps):
        if "id" not in data:
            data["id"] = f"step-{idx}"
        parsed.append(
            StepConfig(
                id=str(data["id"]),
                type=str(data["type"]),
                params=dict(data.get("params", {})),
            )
        )
    return parsed


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config into strongly-typed dataclasses."""

    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    pipeline_block = raw_config.get("pipeline", {})
    raw_steps = pipeline_block.get("steps", [])

    pipeline = PipelineConfig(
        name=str(pipeline_block.get("name", "default-pipeline")),
        mode=str(pipeline_block.get("mode", "sequential")),
        steps=_parse_step_configs(raw_steps),
    )

    return ExperimentConfig(
        video_uri=str(raw_config["video_uri"]),
        transcript_path=str(raw_config["transcript_path"]),
        output_dir=str(raw_config.get("output_dir", "./outputs")),
        video_duration_seconds=raw_config.get("video_duration_seconds"),
        pipeline=pipeline,
        metadata=dict(raw_config.get("metadata", {})),
    )
