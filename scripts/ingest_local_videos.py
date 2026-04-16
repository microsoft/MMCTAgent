"""
Ingest local videos from the 'videos' folder using the MMCT ingestion pipeline.

Discovers video files in the configured videos directory, pairs them with
any co-located transcript (.vtt/.srt) files, and runs the ingestion pipeline
for each video sequentially.

Uses a custom pipeline YAML (scripts/custom_pipeline.yaml) that extends the
default MMCT pipeline with application-specific steps:
  - transcript_upload: uploads SRT transcript to Azure Blob Storage
  - uniform_frames: extracts 1 fps frames and uploads to Azure Blob Storage

These steps are registered at import time via the ``custom_steps`` package
(see scripts/custom_steps/README.md for the custom step pattern).

Uses TOKEN_BROKER_URL from .env for credential resolution via the project's
provider configuration.

Usage:
    python scripts/ingest_local_videos.py [--videos-dir <path>] [--language <lang>] [--verbosity <0|1|2>] [--single]

Examples:
    # Ingest all videos under the default 'videos' folder
    python scripts/ingest_local_videos.py

    # Ingest only the first non-ingested video
    python scripts/ingest_local_videos.py --single

    # Ingest videos from a specific directory with debug logging
    python scripts/ingest_local_videos.py --videos-dir /data/my-videos --verbosity 2
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

from mmct.video_pipeline import IngestionPipeline, Languages
from config.provider_config import get_ingestion_providers

# Register custom ingestion steps (uniform_frames, transcript_upload)
# before the pipeline resolves step types from the YAML config.
import custom_steps  # noqa: F401

# Path to the custom pipeline YAML that includes the custom steps
CUSTOM_PIPELINE_YAML = os.path.join(os.path.dirname(__file__), "custom_pipeline.yaml")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
TRANSCRIPT_EXTENSIONS = [".vtt", ".srt"]

LANGUAGE_MAP = {
    "en": Languages.ENGLISH_UNITED_STATES,
}


def discover_videos(videos_dir: str) -> list[dict]:
    """Walk *videos_dir* and return a list of video entries with metadata.

    Each entry is a dict with keys:
        - video_path: absolute path to the video file
        - transcript_path: path to a co-located transcript (.vtt/.srt), or None
        - metadata: parsed JSON from a co-located metadata file, or {}
        - video_id: the folder/file-based identifier (e.g. '58667')
    """
    entries: list[dict] = []

    for root, _dirs, files in os.walk(videos_dir):
        video_files = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS]
        for vf in video_files:
            video_path = os.path.join(root, vf)
            base = os.path.splitext(vf)[0]
            # Strip common suffixes like '-video' to get the ID stem
            video_id = base.replace("-video", "")

            # Look for a co-located transcript (.vtt or .srt)
            transcript_path = None
            for ext in TRANSCRIPT_EXTENSIONS:
                for pattern in [f"{video_id}-transcript{ext}", f"{video_id}{ext}"]:
                    candidate = os.path.join(root, pattern)
                    if os.path.isfile(candidate):
                        transcript_path = candidate
                        break
                if transcript_path:
                    break

            # Look for a co-located metadata JSON
            metadata = {}
            meta_path = os.path.join(root, f"{video_id}-metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path, "r") as f:
                    metadata = json.load(f)

            entries.append({
                "video_path": video_path,
                "transcript_path": transcript_path,
                "metadata": metadata,
                "video_id": video_id,
            })

    entries.sort(key=lambda e: e["video_id"])
    return entries


async def ingest_video(
    entry: dict,
    providers,
    language: Languages,
    verbosity: int,
) -> bool:
    """Run the ingestion pipeline for a single video entry. Returns True on success."""
    video_path = entry["video_path"]
    video_id = entry["video_id"]
    transcript_path = entry["transcript_path"]
    metadata = entry["metadata"]

    title = metadata.get("title", os.path.basename(video_path))
    url = metadata.get("url")

    logger.info(f"Ingesting '{title}' (id={video_id}) from {video_path}")
    if transcript_path:
        logger.info(f"  Using transcript: {transcript_path}")

    ingestion = IngestionPipeline(
        video_path=video_path,
        video_id=video_id,
        provider=providers,
        language=language,
        transcript_path=transcript_path,
        url=url,
        verbosity=verbosity,
        save_local_report=True,
        pipeline_config_path=CUSTOM_PIPELINE_YAML,
    )

    report = await ingestion.run()

    if report.status == "completed":
        logger.success(
            f"✅ '{title}' ingested in {report.total_duration_seconds:.2f}s"
        )
        return True
    else:
        logger.error(f"❌ '{title}' failed during ingestion")
        return False


def _is_already_ingested(video_id: str, media_dir: str) -> bool:
    """Check if a completed ingestion report exists for *video_id*."""
    report_path = os.path.join(media_dir, f"{video_id}_ip_report.json")
    if not os.path.isfile(report_path):
        return False
    try:
        with open(report_path) as f:
            report = json.load(f)
        return report.get("status") == "completed"
    except (json.JSONDecodeError, OSError):
        return False


async def main_async(videos_dir: str, language: Languages, verbosity: int, single: bool) -> None:
    entries = discover_videos(videos_dir)
    if not entries:
        logger.warning(f"No video files found under '{videos_dir}'")
        return

    media_dir = os.path.join(os.getcwd(), "media")

    if single:
        # Find the first video that hasn't been successfully ingested yet
        remaining = [e for e in entries if not _is_already_ingested(e["video_id"], media_dir)]
        if not remaining:
            logger.info("All videos already ingested — nothing to do")
            return
        entries = remaining[:1]
        logger.info(f"--single flag set, ingesting next non-ingested video: {entries[0]['video_id']}")
    else:
        logger.info(f"Discovered {len(entries)} video(s) in '{videos_dir}'")

    providers = get_ingestion_providers()
    logger.info("Providers initialized successfully")

    succeeded = 0
    failed = 0

    for i, entry in enumerate(entries, start=1):
        logger.info(f"--- [{i}/{len(entries)}] ---")
        try:
            ok = await ingest_video(entry, providers, language, verbosity)
            if ok:
                succeeded += 1
            else:
                failed += 1
        except Exception:
            logger.exception(f"Unhandled error ingesting {entry['video_path']}")
            failed += 1

    logger.info(f"Done — {succeeded} succeeded, {failed} failed out of {len(entries)} video(s)")


def main() -> None:
    default_videos_dir = os.path.join(
        os.path.dirname(__file__), "..", "videos"
    )

    parser = argparse.ArgumentParser(
        description="Ingest local videos from a folder using the MMCT pipeline."
    )
    parser.add_argument(
        "--videos-dir",
        default=default_videos_dir,
        help="Root directory containing video files (default: ./videos)",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=list(LANGUAGE_MAP.keys()),
        help="Source language for transcription (default: en)",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=Progress Bar, 1=Info, 2=Debug (default: 1)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Ingest only the first discovered video (useful for testing)",
    )
    args = parser.parse_args()

    language = LANGUAGE_MAP[args.language]
    asyncio.run(main_async(args.videos_dir, language, args.verbosity, args.single))


if __name__ == "__main__":
    main()
