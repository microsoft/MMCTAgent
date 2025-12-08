"""Configuration loading for pipeline definitions."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""

    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepConfig":
        """Create StepConfig from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            params=data.get("params", {}),
        )


@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline."""

    name: str
    mode: str = "sequential"  # or "parallel"
    steps: List[StepConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create PipelineConfig from dictionary."""
        pipeline_data = data.get("pipeline", data)

        return cls(
            name=pipeline_data.get("name", "unnamed_pipeline"),
            mode=pipeline_data.get("mode", "sequential"),
            steps=[StepConfig.from_dict(step) for step in pipeline_data.get("steps", [])],
        )


def load_pipeline_config(config_path: str) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        PipelineConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
        KeyError: If required fields are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Pipeline config not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty pipeline config: {config_path}")

    return PipelineConfig.from_dict(data)


def get_default_ingestion_config() -> PipelineConfig:
    """
    Get default ingestion pipeline configuration.

    Returns:
        PipelineConfig with default ingestion steps
    """
    return PipelineConfig(
        name="default_ingestion",
        mode="sequential",
        steps=[
            StepConfig(
                id="pre_validation",
                type="ingestion.pre_validation",
                params={
                    "skip_if_exists": True,
                    "require_audio_when_no_transcript": True,
                },
            ),
            StepConfig(
                id="compress",
                type="ingestion.compress",
                params={"max_size_mb": 500, "target_size_mb": 500},
            ),
            StepConfig(
                id="keyframes",
                type="ingestion.keyframes",
                params={"source_step": "compress", "apply_time_offset": False},
            ),
            StepConfig(
                id="transcribe",
                type="ingestion.transcribe",
                params={"video_step": "compress", "apply_time_offset": False},
            ),
            StepConfig(
                id="chapters",
                type="ingestion.chapters",
                params={
                    "transcript_step": "transcribe",
                    "keyframes_step": "keyframes",
                },
            ),
            StepConfig(
                id="embeddings",
                type="ingestion.embeddings",
                params={
                    "chapters_step": "chapters",
                    "keyframes_step": "keyframes",
                },
            ),
            StepConfig(
                id="upload",
                type="ingestion.upload",
                params={
                    "chapters_step": "chapters",
                    "keyframes_step": "keyframes",
                    "embeddings_step": "embeddings",
                },
            ),
            StepConfig(
                id="cleanup",
                type="ingestion.cleanup",
                params={"keep_keyframes": False},
            ),
        ],
    )
