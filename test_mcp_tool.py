"""
Example script to query keyframes using natural language.

This demonstrates how to search for video keyframes using the MCP server.
Run this from the lively directory:
    python run_example_query.py <query> [--video-id VIDEO_ID] [--youtube-url YOUTUBE_URL]

Examples:
    python run_example_query.py "person walking"
    python run_example_query.py "person walking" --video-id abc123
    python run_example_query.py "person walking" --youtube-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from mcp_server.tools.get_visual_timestamps import _get_visual_timestamps_impl

# Load environment variables from lively/.env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run keyframe search example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search for video keyframes using natural language")
    parser.add_argument("query", nargs="*", help="Search query (e.g., 'person walking')")
    parser.add_argument("--video-id", help="Filter results by video ID")
    parser.add_argument("--youtube-url", help="Filter results by YouTube URL (takes precedence over video-id)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top results to return (default: 3)")

    args = parser.parse_args()

    # Get query from arguments or use default
    if args.query:
        query = " ".join(args.query)
    else:
        query = "person walking"
        logger.info(f"No query provided. Using default: '{query}'")

    logger.info("=" * 80)
    logger.info("Lively Visual Search - Query Example")
    logger.info("=" * 80)

    try:
        logger.info(f"Searching for: '{query}'")
        if args.youtube_url:
            logger.info(f"Filtering by YouTube URL: {args.youtube_url}")
        elif args.video_id:
            logger.info(f"Filtering by Video ID: {args.video_id}")
        logger.info(f"Top K: {args.top_k}")
        logger.info("-" * 80)

        # Search for keyframes
        timestamps_str = _get_visual_timestamps_impl(
            query=query,
            top_k=args.top_k,
            video_id=args.video_id,
            youtube_url=args.youtube_url
        )

        if not timestamps_str:
            logger.warning("No keyframes found matching your query.")
            logger.info("Try a different query or check if videos have been indexed.")
            return

        # Convert to list for display
        timestamps = timestamps_str.split(", ")
        logger.info(f"\nFound {len(timestamps)} matching keyframes:\n")
        logger.info(f"📊 Timestamps: {timestamps_str}\n")

        for i, ts in enumerate(timestamps, 1):
            logger.info(f"Result {i}: ⏱ {ts}s")

        logger.info("=" * 80)
        logger.info("Search completed successfully! ✓")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
