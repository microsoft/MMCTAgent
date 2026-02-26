#!/usr/bin/env python3
"""
Local Video Ingestion Script

Ingest a local video file using the temporal graph ingestion pipeline.

Dependencies:
    Requires the mmct package and its dependencies to be installed.
    Run from the project root with venv activated.

Usage:
    python app/ingest_local.py <video_path> --video-id <video_id> --language <language>

Examples:
    python app/ingest_local.py videos/abc123/file.mp4 --video-id abc123 --language ENGLISH_UNITED_STATES
    python app/ingest_local.py /path/to/video.mp4 --video-id my_video_001 --language HINDI

Available Languages:
    ENGLISH_UNITED_STATES, ENGLISH_UNITED_KINGDOM, HINDI, SPANISH, FRENCH, GERMAN,
    CHINESE_SIMPLIFIED, JAPANESE, KOREAN, PORTUGUESE, ITALIAN, RUSSIAN, ARABIC
"""

import argparse
import asyncio
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
    parser.add_argument("video_path", help="Path to the local video file")
    parser.add_argument(
        "--video-id", "-i",
        required=True,
        help="Unique identifier for the video"
    )
    parser.add_argument(
        "--language", "-l",
        required=True,
        help="Language of the video (e.g., ENGLISH_UNITED_STATES, HINDI)"
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

    args = parser.parse_args()

    # Convert language string to enum
    try:
        language = get_language_enum(args.language)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Run ingestion
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
