"""Pipeline runner for orchestrating step execution."""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .steps.base import StepContext, StepResult
from .config.schemas import PipelineConfig
from .steps.registry import get_step_class


@dataclass
class StepExecutionRecord:
    """Record of a step's execution."""

    step_id: str
    step_type: str
    status: str  # "completed", "failed", "skipped"
    duration_seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class PipelineReport:
    """Report of pipeline execution."""

    pipeline_name: str
    total_duration_seconds: float
    status: str  # "completed", "failed", "partial"
    steps: List[StepExecutionRecord] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        completed = sum(1 for s in self.steps if s.status == "completed")
        failed = sum(1 for s in self.steps if s.status == "failed")

        return (
            f"Pipeline '{self.pipeline_name}' {self.status}\n"
            f"  Duration: {self.total_duration_seconds:.2f}s\n"
            f"  Steps: {completed} completed, {failed} failed, {len(self.steps)} total"
        )

    def save(self, output_path: str) -> None:
        """Save report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class PipelineRunner:
    """
    Orchestrates execution of pipeline steps.

    Handles sequential execution, result tracking, error handling,
    and report generation.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        context: StepContext,
    ):
        """
        Initialize the pipeline runner.

        Args:
            pipeline_config: Pipeline configuration with steps to execute
            context: Shared context for all steps
        """
        self.pipeline_config = pipeline_config
        self.context = context
        self.logger = context.logger
        self.execution_records: List[StepExecutionRecord] = []

    async def run(self) -> PipelineReport:
        """
        Execute the pipeline.

        Returns:
            PipelineReport with execution details

        Raises:
            Exception: If a step fails and error handling doesn't recover
        """
        pipeline_start = time.time()
        pipeline_status = "completed"

        if self.context.verbosity >= 1:
            self.logger.info(f"Starting pipeline: {self.pipeline_config.name}")
            self.logger.info(f"Mode: {self.pipeline_config.mode}")
            self.logger.info(f"Steps: {len(self.pipeline_config.steps)}")

        try:
            steps_iterable = self.pipeline_config.steps

            # If verbosity is 0, wrap with tqdm
            if self.context.verbosity == 0:
                from tqdm import tqdm

                steps_iterable = tqdm(
                    self.pipeline_config.steps,
                    desc=f"Ingesting {self.context.video_id[:8]}...",
                    unit="step",
                )

            for idx, step_cfg in enumerate(steps_iterable, 1):

                if self.context.verbosity >= 1:
                    self.logger.info(
                        f"[{idx}/{len(self.pipeline_config.steps)}] Executing step: {step_cfg.id} ({step_cfg.type})"
                    )
                elif self.context.verbosity == 0:
                    if hasattr(steps_iterable, "set_description"):
                        steps_iterable.set_description(f"Step: {step_cfg.id}")

                try:
                    # Get step class from registry
                    StepClass = get_step_class(step_cfg.type)

                    # Instantiate step
                    step = StepClass(step_id=step_cfg.id, params=step_cfg.params)

                    # Execute step
                    step_start = time.time()
                    result = await step.run(self.context)
                    step_duration = time.time() - step_start

                    # Store outputs in data store
                    for key, value in result.outputs.items():
                        self.context.data_store.set(step_cfg.id, key, value)

                    # Record execution
                    record = StepExecutionRecord(
                        step_id=step_cfg.id,
                        step_type=step_cfg.type,
                        status="completed",
                        duration_seconds=step_duration,
                        metrics=result.metrics,
                        artifacts=result.artifacts,
                    )
                    self.execution_records.append(record)

                    if self.context.verbosity >= 1:
                        self.logger.info(f"Step {step_cfg.id} completed in {step_duration:.2f}s")

                    # Check if step requested pipeline termination
                    should_continue = result.outputs.get("should_continue", True)
                    if not should_continue:
                        msg = f"Step {step_cfg.id} requested pipeline termination. Skipping remaining steps."
                        if self.context.verbosity >= 1:
                            self.logger.info(msg)
                        pipeline_status = "completed"
                        break

                except Exception as e:
                    step_duration = time.time() - step_start if "step_start" in locals() else 0.0

                    self.logger.exception(f"Step {step_cfg.id} failed: {e}")

                    # Record failure
                    record = StepExecutionRecord(
                        step_id=step_cfg.id,
                        step_type=step_cfg.type,
                        status="failed",
                        duration_seconds=step_duration,
                        error=str(e),
                    )
                    self.execution_records.append(record)

                    pipeline_status = "failed"

                    # Re-raise to stop pipeline execution
                    raise

        except Exception as e:
            if self.context.verbosity >= 1:
                self.logger.exception(f"Pipeline {self.pipeline_config.name} failed: {e}")
            pipeline_status = "failed"
            raise

        finally:
            # Generate report
            total_duration = time.time() - pipeline_start

            report = PipelineReport(
                pipeline_name=self.pipeline_config.name,
                total_duration_seconds=total_duration,
                status=pipeline_status,
                steps=self.execution_records,
            )

            self.logger.info(report.summary())

            # Save report if output_dir is available and enabled
            if self.context.output_dir and self.context.save_local_report:
                report_path = os.path.join(
                    self.context.output_dir,
                    f"{self.context.video_id}_ip_report.json",
                )
                report.save(report_path)
                self.logger.info(f"Pipeline report saved to: {report_path}")
            elif not self.context.save_local_report:
                self.logger.info("Report saving disabled by configuration.")

        return report
