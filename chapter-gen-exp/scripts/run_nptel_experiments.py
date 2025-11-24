"""Batch runner that executes `run_experiment.py` for many configs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def list_configs(config_dir: Path, include: Sequence[str] | None) -> Iterable[Path]:
    configs = sorted(config_dir.glob("*.yaml"))
    if not include:
        return configs
    include_set = set(include)
    return [config for config in configs if config.stem in include_set]


def read_output_dir(config_path: Path) -> Path | None:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    output_dir = data.get("output_dir")
    if not output_dir:
        return None
    candidate = Path(output_dir)
    return candidate if candidate.is_absolute() else ROOT / candidate


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_command(config_path: Path, report_path: Path | None) -> list[str]:
    cmd = [sys.executable, str(RUN_EXPERIMENT), "--config", str(config_path)]
    if report_path is not None:
        ensure_parent(report_path)
        cmd.extend(["--report", str(report_path)])
    return cmd


def run_command(cmd: list[str], dry_run: bool) -> int:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return 0
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chapter-generation experiments in batch")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/generated/nptel"),
        help="Directory that holds YAML configs",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Optional base directory for reports (defaults to config's output_dir)",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Optional list of config stems to run (e.g., video IDs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = resolve_path(args.config_dir)
    report_dir = resolve_path(args.report_dir) if args.report_dir else None

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)

    configs = list(list_configs(config_dir, args.include))
    if not configs:
        print("No config files found. Nothing to run.")
        return

    successes = []
    failures = []

    for config_path in configs:
        output_dir = read_output_dir(config_path)
        default_report = output_dir / "report.json" if output_dir else None
        report_path = (report_dir / config_path.stem / "report.json") if report_dir else default_report
        cmd = build_command(config_path, report_path)
        print(f"Running {config_path.stem}...")
        ret_code = run_command(cmd, args.dry_run)
        if ret_code == 0:
            successes.append(config_path.stem)
        else:
            failures.append((config_path.stem, ret_code))

    print(f"Completed {len(successes)} experiment(s)")
    if successes:
        print(" - " + ", ".join(successes))
    if failures:
        print("Failures:")
        for name, code in failures:
            print(f" - {name}: exit code {code}")


if __name__ == "__main__":
    main()
