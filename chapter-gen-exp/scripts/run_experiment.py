"""CLI entry point for running experimentation pipelines."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - runtime path guard
    sys.path.append(str(ROOT))

from pipelines.config.schemas import load_experiment_config
from pipelines.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a chapter-generation experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/sample_config.yaml"),
        help="Path to the experiment YAML file",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    runner = PipelineRunner(config)
    report = runner.run()

    print(f"Pipeline '{report.pipeline_name}' finished in {report.total_duration_seconds:.2f}s")
    for record in report.steps:
        print(
            f" - Step {record.step_id} ({record.step_type}): {record.duration_seconds:.2f}s, "
            f"metrics={record.metrics}"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(
                {
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
                },
                handle,
                indent=2,
            )
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
