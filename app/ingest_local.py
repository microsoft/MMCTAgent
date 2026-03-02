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

Examples:
    python app/ingest_local.py videos/abc123/file.mp4 --video-id abc123 --language ENGLISH_UNITED_STATES
    python app/ingest_local.py --batch my_videos.json
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

    args = parser.parse_args()

    if args.batch:
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
