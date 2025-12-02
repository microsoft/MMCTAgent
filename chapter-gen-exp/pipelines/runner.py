"""High-level orchestration helpers for experimentation pipelines."""
from __future__ import annotations

import json
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
        self._completed_steps: Dict[str, Dict[str, Dict[str, object]]] = {}
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
            video_id=config.video_id,
            metadata=config.metadata,
            data_store=self._data_store,
            video_metadata=self._video_metadata,
        )

        self._load_previous_results()

    def run(self) -> PipelineReport:
        start = time.perf_counter()
        records: List[StepExecutionRecord] = []

        if self.config.pipeline.mode != "sequential":
            raise NotImplementedError(
                f"Execution mode '{self.config.pipeline.mode}' is not supported yet"
            )

        for step_cfg in self.config.pipeline.steps:
            if step_cfg.id in self._completed_steps:
                cached = self._completed_steps[step_cfg.id]
                logger.info(
                    "[runner] Skipping step '%s' (%s); already completed",
                    step_cfg.id,
                    step_cfg.type,
                )
                resumed_metrics: Dict[str, float] = {}
                for key, value in (cached.get("metrics") or {}).items():
                    try:
                        resumed_metrics[key] = float(value)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        logger.debug(
                            "[runner] Unable to coerce metric '%s' for step '%s'",
                            key,
                            step_cfg.id,
                        )

                resumed_artifacts = {
                    k: str(v) for k, v in (cached.get("artifacts") or {}).items()
                }

                records.append(
                    StepExecutionRecord(
                        step_id=step_cfg.id,
                        step_type=step_cfg.type,
                        duration_seconds=0.0,
                        metrics=resumed_metrics,
                        artifacts=resumed_artifacts,
                    )
                )
                continue

            step_cls = get_step_class(step_cfg.type)
            step = step_cls(step_cfg.id, step_cfg.params)
            logger.info(
                "[runner] Starting step '{}' ({})",
                step_cfg.id,
                step_cfg.type,
            )
            step_start = time.perf_counter()
            try:
                result = step.run(self._context)
            except Exception:
                step_duration = time.perf_counter() - step_start
                logger.exception(
                    "[runner] Step '%s' (%s) failed after %.2fs",
                    step_cfg.id,
                    step_cfg.type,
                    step_duration,
                )
                total_duration = time.perf_counter() - start
                report = self._build_report(records, total_duration)
                self._write_report(report)
                raise

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
        report = self._build_report(records, total_duration)
        self._write_report(report)
        logger.info(
            "[runner] Pipeline '{}' finished in {:.2f}s",
            self.config.pipeline.name,
            total_duration,
        )
        return report

    def _load_previous_results(self) -> None:
        """Load prior step outputs/metadata if a report already exists."""

        report_path = self.output_dir / "report.json"
        if not report_path.exists():
            return

        try:
            with report_path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[runner] Failed to parse existing report at %s: %s", report_path, exc)
            return

        previous_pipeline = report.get("pipeline")
        if previous_pipeline and previous_pipeline != self.config.pipeline.name:
            logger.warning(
                "[runner] Existing report references pipeline '%s'; expected '%s'. Skipping resume.",
                previous_pipeline,
                self.config.pipeline.name,
            )
            return

        outputs = report.get("outputs", {}) or {}
        for step_id, payload in outputs.items():
            bucket = self._data_store.namespace(step_id)
            bucket.update(payload)

        steps = report.get("steps", []) or []
        cached: Dict[str, Dict[str, Dict[str, object]]] = {}
        for entry in steps:
            step_id = entry.get("id")
            if not step_id:
                continue
            cached[step_id] = {
                "type": entry.get("type", ""),
                "metrics": entry.get("metrics", {}) or {},
                "artifacts": entry.get("artifacts", {}) or {},
            }

        if cached:
            self._completed_steps = cached
            logger.info(
                "[runner] Loaded %d completed steps from %s; resume mode enabled",
                len(cached),
                report_path,
            )

    def _build_report(
        self,
        records: List[StepExecutionRecord],
        total_duration: float,
    ) -> PipelineReport:
        return PipelineReport(
            pipeline_name=self.config.pipeline.name,
            total_duration_seconds=total_duration,
            steps=list(records),
            outputs_snapshot={key: value for key, value in self._data_store.as_dict().items()},
        )

    def _write_report(self, report: PipelineReport) -> None:
        report_path = self.output_dir / "report.json"
        payload = {
            "pipeline": report.pipeline_name,
            "duration_seconds": report.total_duration_seconds,
            "steps": [
                {
                    "id": record.step_id,
                    "type": record.step_type,
                    "metrics": record.metrics,
                    "artifacts": record.artifacts,
                }
                for record in report.steps
            ],
            "outputs": report.outputs_snapshot,
        }

        try:
            with report_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            logger.info("[runner] Snapshot report written to %s", report_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("[runner] Failed to write report at %s: %s", report_path, exc)
