"""
Example usage of the MCP Server visual search tools.

This demonstrates how to:
1. Search for keyframes using natural language queries
2. Get visual summaries with timestamps
3. Filter results by video ID
"""

import asyncio
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from tools import search_keyframes, get_visual_timestamps

# Load environment variables from lively/.env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_keyframe_search():
    """Example: Search for keyframes matching a query."""
    logger.info("=" * 80)
    logger.info("Example: Keyframe Search")
    logger.info("=" * 80)

    try:
        # Search for keyframes matching a query
        query = "person walking"

        results = search_keyframes(
            query=query,
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            top_k=5
        )

        logger.info(f"\nQuery: '{results['query']}'")
        logger.info(f"Total results: {results['total_results']}\n")

        for i, keyframe in enumerate(results['keyframes'], 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Timestamp: {keyframe['timestamp_seconds']:.2f}s")
            logger.info(f"  Video ID: {keyframe['video_id']}")
            logger.info(f"  Filename: {keyframe['keyframe_filename']}")
            logger.info(f"  Blob URL: {keyframe['blob_url']}")
            logger.info(f"  YouTube URL: {keyframe.get('youtube_url', 'N/A')}")
            logger.info(f"  Search Score: {keyframe['search_score']:.4f}")
            logger.info("")

    except Exception as e:
        logger.error(f"Failed to search keyframes: {e}", exc_info=True)


def example_visual_timestamps():
    """Example: Get just timestamps for a query."""
    logger.info("=" * 80)
    logger.info("Example: Timestamp Extraction")
    logger.info("=" * 80)

    try:
        query = "car driving on road"

        timestamps = get_visual_timestamps(
            query=query,
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            top_k=5
        )

        logger.info(f"\nQuery: '{query}'")
        logger.info(f"Found {len(timestamps)} timestamps\n")

        for i, ts in enumerate(timestamps, 1):
            logger.info(f"{i}. {ts['timestamp_seconds']:.2f}s - {ts['keyframe_filename']}")

    except Exception as e:
        logger.error(f"Failed to get timestamps: {e}", exc_info=True)


def example_filtered_search():
    """Example: Search filtered by video ID."""
    logger.info("=" * 80)
    logger.info("Example: Filtered Search by Video ID")
    logger.info("=" * 80)

    try:
        query = "outdoor scene"
        video_id = "your_video_id_here"  # Replace with actual video ID

        results = search_keyframes(
            query=query,
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            top_k=5,
            video_id=video_id
        )

        logger.info(f"\nQuery: '{query}' (filtered by video: {video_id})")
        logger.info(f"Total results: {results['total_results']}\n")

        for keyframe in results['keyframes']:
            logger.info(f"  {keyframe['timestamp_seconds']:.2f}s - {keyframe['keyframe_filename']}")

    except Exception as e:
        logger.error(f"Failed to search with filter: {e}", exc_info=True)


def main():
    """Run all examples."""
    logger.info("Starting MCP Server Visual Search Examples")

    # Example 1: Keyframe search
    example_keyframe_search()

    # Example 2: Timestamps only
    example_visual_timestamps()

    # Example 3: Filtered search
    # example_filtered_search()  # Uncomment and provide video_id to test

    logger.info("Examples completed!")


if __name__ == "__main__":
    main()
