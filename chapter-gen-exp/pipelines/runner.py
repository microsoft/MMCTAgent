"""High-level orchestration helpers for experimentation pipelines."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from .config.schemas import ExperimentConfig
from .steps import builtins  # noqa: F401  # ensures built-in registrations
from .steps.base import StepContext, StepDataStore
from .steps.registry import get_step_class
from .utils.video import VideoMetadata, extract_video_metadata


@dataclass
class StepExecutionRecord:
    step_id: str
    step_type: str
    duration_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineReport:
    pipeline_name: str
    total_duration_seconds: float
    steps: List[StepExecutionRecord]
    outputs_snapshot: Dict[str, Dict[str, object]]


class PipelineRunner:
    """Simple sequential runner with hooks for future parallel modes."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._data_store = StepDataStore()
        self._video_metadata: Optional[VideoMetadata] = None
        video_duration = config.video_duration_seconds

        try:
            metadata = extract_video_metadata(config.video_uri)
            self._video_metadata = metadata
            video_duration = metadata.duration_seconds
        except Exception as exc:
            if video_duration is None:
                raise RuntimeError(
                    f"Unable to infer duration from video '{config.video_uri}'."
                ) from exc

        self._context = StepContext(
            video_uri=config.video_uri,
            transcript_path=config.transcript_path,
            output_dir=self.output_dir,
            video_duration_seconds=video_duration,
            metadata=config.metadata,
            data_store=self._data_store,
            video_metadata=self._video_metadata,
        )

    def run(self) -> PipelineReport:
        start = time.perf_counter()
        records: List[StepExecutionRecord] = []

        if self.config.pipeline.mode != "sequential":
            raise NotImplementedError(
                f"Execution mode '{self.config.pipeline.mode}' is not supported yet"
            )

        for step_cfg in self.config.pipeline.steps:
            step_cls = get_step_class(step_cfg.type)
            step = step_cls(step_cfg.id, step_cfg.params)
            logger.info(
                "[runner] Starting step '{}' ({})",
                step_cfg.id,
                step_cfg.type,
            )
            step_start = time.perf_counter()
            result = step.run(self._context)
            step_duration = time.perf_counter() - step_start

            bucket = self._data_store.namespace(step_cfg.id)
            bucket.update(result.produced)

            records.append(
                StepExecutionRecord(
                    step_id=step_cfg.id,
                    step_type=step_cfg.type,
                    duration_seconds=step_duration,
                    metrics=result.metrics,
                    artifacts=result.artifacts,
                )
            )

        total_duration = time.perf_counter() - start
        logger.info(
            "[runner] Pipeline '{}' finished in {:.2f}s",
            self.config.pipeline.name,
            total_duration,
        )
        return PipelineReport(
            pipeline_name=self.config.pipeline.name,
            total_duration_seconds=total_duration,
            steps=records,
            outputs_snapshot={key: value for key, value in self._data_store.as_dict().items()},
        )
