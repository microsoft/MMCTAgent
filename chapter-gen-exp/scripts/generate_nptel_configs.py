"""Utility to auto-generate experiment YAML files for NPTEL samples."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PIPELINE_STEPS = [
    {
        "id": "scene_chunks",
        "type": "video.chunk.scene",
        "params": {
            "detector": "adaptive",
            "threshold": 3,
            "min_scene_length": 10,
            "max_scenes": 1000,
        },
    },
    {
        "id": "scene_chunk_alignment",
        "type": "video.chunk.align-transcript",
        "params": {
            "chunks_step": "scene_chunks",
        },
    },
    {
        "id": "of_scene_frames",
        "type": "frames.optical-flow",
        "params": {
            "motion_threshold": 0.8,
            "sample_fps": 1.0,
            "max_frame_width": 720,
            "max_frames": 150,
            "max_frames_per_chunk": 10,
            "output_subdir": "frames-optical",
            "chunks_step": "scene_chunk_alignment",
        },
    },
    {
        "id": "llm_chapters",
        "type": "chapters.scene-llm",
        "params": {
            "chunks_step": "scene_chunk_alignment",
            "frames_step": "of_scene_frames",
            "batch_size": 5,
            "max_parallel_requests": 5,
            "max_frames_per_chapter": 10,
        },
    },
    {
        "id": "enriched_chapters",
        "type": "chapters.context-enrich",
        "params": {
            "chapters_step": "llm_chapters",
            "context_window": 4,
            "object_enrichment": {
                "enabled": True,
                "max_active_context": 10,
                "min_screen_time_seconds": 8.0,
                "min_chunk_occurrences": 2,
            },
        },
    },
    {
        "id": "frame_blob_export",
        "type": "export.frame-blob-upload",
        "params": {
            "frames_step": "of_scene_frames",
            "container_name": "kv-new-chapter-gen-frames",
            "index_name": "keyframes-new-chapter-gen",
        },
    },
    {
        "id": "chapter_index_export",
        "type": "export.chapter-search-index",
        "params": {
            "chapters_step": "enriched_chapters",
            "index_name": "chapters-new-chapter-gen",
        },
    },
    {
        "id": "object_index_export",
        "type": "export.object-collection-index",
        "params": {
            "source_step": "enriched_chapters",
            "index_name": "objects-new-chapter-gen",
        },
    },
]


def repo_relative(path: Path) -> str:
    """Return a path relative to the repo root when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    """Interpret relative paths from the repository root."""
    return path if path.is_absolute() else ROOT / path


def load_videos(video_dir: Path) -> Iterable[Path]:
    return sorted(video_dir.glob("*.mp4"))


def build_config(video_path: Path, transcript_path: Path, run_output_root: Path) -> dict:
    video_id = video_path.stem
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    output_dir = run_output_root / video_id

    return {
        "video_uri": repo_relative(video_path),
        "transcript_path": repo_relative(transcript_path),
        "output_dir": repo_relative(output_dir),
        "metadata": {
            "experiment": "nptel auto-generated",
            "notes": "chapter and frames export",
            "youtube_url": youtube_url,
        },
        "pipeline": {
            "name": f"NPTEL ingestion for {video_id}",
            "mode": "sequential",
            "steps": copy.deepcopy(DEFAULT_PIPELINE_STEPS),
        },
    }


def write_yaml(config: dict, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {destination}")

    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment YAML files for NPTEL videos")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("samples/videos/nptel"),
        help="Directory containing .mp4 files",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("samples/transcripts/nptel"),
        help="Directory containing transcript .en.srt files",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/generated/nptel"),
        help="Where to write generated YAML configs",
    )
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=Path("outputs/nptel"),
        help="Base directory referenced by output_dir field inside each config",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing YAML files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = resolve_path(args.videos_dir)
    transcripts_dir = resolve_path(args.transcripts_dir)
    config_dir = resolve_path(args.config_dir)
    run_output_dir = resolve_path(args.run_output_dir)

    if not videos_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {videos_dir}")
    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcript directory not found: {transcripts_dir}")

    written = []
    skipped = []
    missing_transcripts = []

    for video_path in load_videos(videos_dir):
        transcript_path = transcripts_dir / f"{video_path.stem}.en.srt"
        if not transcript_path.exists():
            missing_transcripts.append(video_path.stem)
            continue

        config = build_config(video_path, transcript_path, run_output_dir)
        destination = config_dir / f"{video_path.stem}.yaml"
        try:
            write_yaml(config, destination, overwrite=args.overwrite)
        except FileExistsError:
            skipped.append(video_path.stem)
        else:
            written.append(video_path.stem)

    print(f"Generated {len(written)} config(s) in {config_dir}")
    if written:
        print(" - " + ", ".join(sorted(written)))
    if skipped:
        print(f"Skipped existing configs (use --overwrite to regenerate): {', '.join(sorted(skipped))}")
    if missing_transcripts:
        print(f"Missing transcripts for: {', '.join(sorted(missing_transcripts))}")


if __name__ == "__main__":
    main()
