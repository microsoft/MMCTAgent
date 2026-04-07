#!/usr/bin/env python3
"""
Local Video Ingestion Script

Ingest a local video file using the temporal graph ingestion pipeline.

Dependencies:
    Requires the mmct package and its dependencies to be installed.
    Run from the project root with venv activated.

Usage:
    # Single video:
    python app/ingest_local.py <video_path> --video-id <video_id> --language <language>

    # Batch mode (reads videos from a JSON manifest):
    python app/ingest_local.py --batch batch_manifest.json

    # Folder mode (auto-discovers videos in a playlist directory):
    python app/ingest_local.py --folder playlist_PLZ2ps__7DhBaDccbZRgiU1sHX2gZrQ-XT

    # Retry only specific video IDs from a folder:
    python app/ingest_local.py --folder playlist_PLZ2ps__7DhBaDccbZRgiU1sHX2gZrQ-XT --only id1,id2,id3

Batch JSON format:
    [
        {
            "video_path": "videos/abc123/file.mp4",
            "video_id": "abc123",
            "language": "ENGLISH_UNITED_STATES",
            "url": "https://www.youtube.com/watch?v=abc123"   // optional
        },
        ...
    ]

Folder mode:
    Scans videos/<folder>/ for subdirectories containing file.mp4.
    Each subdirectory name is used as the video_id.
    YouTube URLs are auto-generated from video_id.

Examples:
    python app/ingest_local.py videos/abc123/file.mp4 --video-id abc123 --language ENGLISH_UNITED_STATES
    python app/ingest_local.py --batch my_videos.json
    python app/ingest_local.py --folder playlist_PLZ2ps__7DhBaDccbZRgiU1sHX2gZrQ-XT
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from mmct.video_pipeline import IngestionPipeline, Languages
from mmct.video_pipeline.utils.helper import remove_file
from app.config import get_ingestion_provider

# Path to temporal graph ingestion pipeline config
TEMPORAL_GRAPH_PIPELINE_CONFIG = project_root / \
    "mmct/video_pipeline/core/ingestion/experiments/temporal_graph_ingestion.yaml"


async def ingest_local_video(
    video_path: str,
    video_id: str,
    language: Languages,
    transcript_path: str = None,
    url: str = None,
    verbosity: int = 2,
    playlist_id: str = None,
    playlist_order: int = None,
) -> dict:
    """
    Ingest a local video file using the temporal graph pipeline.

    Args:
        video_path: Path to the local video file
        video_id: Unique identifier for the video
        language: Language of the video (Languages enum)
        transcript_path: Optional path to existing transcript file
        url: Optional URL associated with the video
        verbosity: Logging verbosity (0=quiet, 1=info, 2=debug)
        playlist_id: Optional playlist ID this video belongs to
        playlist_order: Optional 1-based position within the playlist

    Returns:
        Dict with ingestion results
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Starting ingestion for video: {video_id}")
    logger.info(f"Video path: {video_path}")
    logger.info(f"Language: {language}")
    logger.info(f"Pipeline config: {TEMPORAL_GRAPH_PIPELINE_CONFIG}")

    try:
        # Get ingestion provider configuration
        provider = get_ingestion_provider()

        # Create IngestionPipeline with temporal graph pipeline config
        pipeline = IngestionPipeline(
            video_path=video_path,
            video_id=video_id,
            language=language,
            transcript_path=transcript_path,
            url=url,
            provider=provider,
            pipeline_config_path=str(TEMPORAL_GRAPH_PIPELINE_CONFIG),
            verbosity=verbosity,
            playlist_id=playlist_id,
            playlist_order=playlist_order,
        )
        await pipeline.run()

        logger.info(f"Successfully ingested video {video_id}")

        return {
            "message": "success",
            "video_id": video_id,
            "pipeline": "temporal_graph_ingestion",
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        # Clean up temporary files created during ingestion
        await remove_file(video_id=video_id)


def get_language_enum(language_str: str) -> Languages:
    """Convert language string to Languages enum."""
    try:
        return Languages[language_str.upper()]
    except KeyError:
        available = [lang.name for lang in Languages]
        raise ValueError(
            f"Invalid language: {language_str}\n"
            f"Available languages: {', '.join(available)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a local video using temporal graph pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("video_path", nargs="?", default=None,
                        help="Path to the local video file (omit when using --batch)")
    parser.add_argument(
        "--video-id", "-i",
        default=None,
        help="Unique identifier for the video"
    )
    parser.add_argument(
        "--language", "-l",
        default="ENGLISH_UNITED_STATES",
        help="Language of the video (default: ENGLISH_UNITED_STATES)"
    )
    parser.add_argument(
        "--transcript", "-t",
        default=None,
        help="Optional path to existing transcript file (.srt)"
    )
    parser.add_argument(
        "--url", "-u",
        default=None,
        help="Optional URL associated with the video"
    )
    parser.add_argument(
        "--verbosity", "-v",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=quiet, 1=info, 2=debug (default: 2)"
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        metavar="JSON_FILE",
        help="Path to a JSON manifest for batch ingestion"
    )
    parser.add_argument(
        "--folder", "-f",
        default=None,
        metavar="FOLDER_NAME",
        help="Folder inside videos/ to scan for video subdirectories (e.g. playlist_PLZ2ps__7DhBaDccbZRgiU1sHX2gZrQ-XT)"
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="VIDEO_IDS",
        help="Comma-separated list of video IDs to process (use with --folder to retry specific videos)"
    )
    parser.add_argument(
        "--playlist-id",
        default=None,
        metavar="PLAYLIST_ID",
        help="Playlist ID this video (or folder of videos) belongs to"
    )
    parser.add_argument(
        "--playlist-order",
        type=int,
        default=None,
        metavar="ORDER",
        help="1-based position of the video within its playlist (single video mode only)"
    )

    args = parser.parse_args()

    if args.batch and args.folder:
        print("Error: --batch and --folder cannot be used together")
        sys.exit(1)

    if args.folder:
        # ----------------------------------------------------------
        # Folder mode: auto-discover videos in a playlist directory
        # ----------------------------------------------------------
        videos_dir = project_root / "videos" / args.folder
        if not videos_dir.is_dir():
            print(f"Error: Folder not found: {videos_dir}")
            sys.exit(1)

        # Find all subdirectories containing file.mp4
        entries = sorted([
            d for d in videos_dir.iterdir()
            if d.is_dir() and (d / "file.mp4").exists()
        ], key=lambda d: d.name)

        # Filter to specific video IDs if --only is provided
        if args.only:
            only_ids = set(args.only.split(","))
            entries = [d for d in entries if d.name in only_ids]
            missing = only_ids - {d.name for d in entries}
            if missing:
                print(f"Warning: {len(missing)} video IDs not found in folder: {', '.join(sorted(missing))}")

        if not entries:
            print(f"Error: No video subdirectories with file.mp4 found in {videos_dir}")
            sys.exit(1)

        try:
            language = get_language_enum(args.language)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print(f"Found {len(entries)} videos in {videos_dir.name}")
        print(f"Language: {language.name}")
        if args.playlist_id:
            print(f"Playlist ID: {args.playlist_id}")

        async def run_folder():
            results = []
            for idx, video_dir in enumerate(entries, 1):
                vid = video_dir.name
                vpath = str(video_dir / "file.mp4")
                url = f"https://www.youtube.com/watch?v={vid}"

                print(f"\n{'='*60}")
                print(f"[{idx}/{len(entries)}] Ingesting video: {vid}")
                print(f"{'='*60}")
                try:
                    await ingest_local_video(
                        video_path=vpath,
                        video_id=vid,
                        language=language,
                        url=url,
                        verbosity=args.verbosity,
                        playlist_id=args.playlist_id,
                        playlist_order=idx if args.playlist_id else None,
                    )
                    results.append((vid, "success"))
                    print(f"  ✓ {vid} ingested successfully")
                except Exception as e:
                    results.append((vid, f"failed: {e}"))
                    print(f"  ✗ {vid} failed: {e}")

            print(f"\n{'='*60}")
            print("Folder ingestion summary:")
            print(f"{'='*60}")
            ok = sum(1 for _, s in results if s == "success")
            fail = len(results) - ok
            for vid, status in results:
                marker = "✓" if status == "success" else "✗"
                print(f"  {marker} {vid}: {status}")
            print(f"\nTotal: {ok} succeeded, {fail} failed out of {len(results)}")

        asyncio.run(run_folder())

    elif args.batch:
        # ----------------------------------------------------------
        # Batch mode: read video list from JSON file
        # ----------------------------------------------------------
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"Error: Batch file not found: {batch_path}")
            sys.exit(1)

        try:
            with open(batch_path, "r") as f:
                batch_entries = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {batch_path}: {e}")
            sys.exit(1)

        if not isinstance(batch_entries, list):
            print("Error: Batch JSON must be a list of objects")
            sys.exit(1)

        async def run_batch():
            results = []
            for idx, entry in enumerate(batch_entries):
                vid = entry.get("video_id")
                vpath = entry.get("video_path")
                lang_str = entry.get("language", "ENGLISH_UNITED_STATES")

                if not vid or not vpath:
                    print(f"  ✗ Entry {idx}: missing video_id or video_path, skipping")
                    results.append((vid or f"entry_{idx}", "skipped: missing fields"))
                    continue

                try:
                    lang = get_language_enum(lang_str)
                except ValueError as e:
                    results.append((vid, f"skipped: {e}"))
                    continue

                print(f"\n{'='*60}")
                print(f"Ingesting video: {vid}")
                print(f"{'='*60}")
                try:
                    await ingest_local_video(
                        video_path=str(vpath),
                        video_id=vid,
                        language=lang,
                        url=entry.get("url"),
                        verbosity=args.verbosity,
                        playlist_id=entry.get("playlist_id") or args.playlist_id,
                        playlist_order=entry.get("playlist_order"),
                    )
                    results.append((vid, "success"))
                    print(f"  ✓ {vid} ingested successfully")
                except Exception as e:
                    results.append((vid, f"failed: {e}"))
                    print(f"  ✗ {vid} failed: {e}")

            print(f"\n{'='*60}")
            print("Batch ingestion summary:")
            print(f"{'='*60}")
            for vid, status in results:
                print(f"  {vid}: {status}")

        asyncio.run(run_batch())
    else:
        # ----------------------------------------------------------
        # Single video mode
        # ----------------------------------------------------------
        if not args.video_path:
            print("Error: video_path is required in single video mode (or use --batch)")
            sys.exit(1)
        if not args.video_id:
            print("Error: --video-id is required in single video mode")
            sys.exit(1)

        try:
            language = get_language_enum(args.language)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        try:
            result = asyncio.run(
                ingest_local_video(
                    video_path=args.video_path,
                    video_id=args.video_id,
                    language=language,
                    transcript_path=args.transcript,
                    url=args.url,
                    verbosity=args.verbosity,
                    playlist_id=args.playlist_id,
                    playlist_order=args.playlist_order,
                )
            )
            print(f"\nIngestion complete!")
            print(f"  Video ID: {result['video_id']}")
            print(f"  Pipeline: {result['pipeline']}")
            print(f"  Status: {result['message']}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Ingestion failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
