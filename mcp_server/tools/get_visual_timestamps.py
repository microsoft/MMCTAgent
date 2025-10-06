"""
Simplified tool to get timestamps from keyframes in Azure AI Search.

This is a wrapper around search_keyframes that returns just the timestamps.
"""

import os
from typing import Optional
from ..server import mcp
from .visual_search_client import search_keyframes


def _get_visual_timestamps_impl(
    query: str,
    top_k: int = 3,
    video_id: Optional[str] = None,
    youtube_url: Optional[str] = None
) -> str:
    """
    Implementation of get_visual_timestamps that can be called directly.

    Args:
        query: Text query describing what to search for (e.g., "person walking", "car driving")
        top_k: Number of top results to return (default: 3)
        video_id: Optional video ID to filter results to a specific video
        youtube_url: Optional YouTube URL to filter results. Takes precedence over video_id if both provided.

    Returns:
        Comma-separated string of timestamps in seconds (e.g., "5.0, 12.5, 18.0")
    """
    # Load config from environment
    search_endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT")
    index_name = os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index")
    search_api_key = os.getenv("SEARCH_API_KEY")

    if not search_endpoint:
        return "Error: SEARCH_SERVICE_ENDPOINT not configured in environment"

    # Get search results
    try:
        results = search_keyframes(
            query=query,
            search_endpoint=search_endpoint,
            index_name=index_name,
            search_api_key=search_api_key,
            top_k=top_k,
            video_id=video_id,
            youtube_url=youtube_url
        )

        # Extract timestamp values and format as comma-separated string
        timestamps = [str(keyframe["timestamp_seconds"]) for keyframe in results["keyframes"]]

        if not timestamps:
            return f"No keyframes found for query: '{query}'"

        return ", ".join(timestamps)

    except Exception as e:
        return f"Error searching keyframes: {str(e)}"


@mcp.tool(
    name="get_visual_timestamps",
    description=(
        "Get top timestamps of keyframes matching a text query. "
        "Returns a comma-separated string of timestamps in seconds (e.g., '5.0, 12.5, 18.0'). "
        "Use this to find specific moments in videos based on visual content description. "
        "If both youtube_url and video_id are provided, youtube_url takes precedence for filtering."
    )
)
def get_visual_timestamps(
    query: str,
    top_k: int = 3,
    video_id: Optional[str] = None,
    youtube_url: Optional[str] = None
) -> str:
    """
    Get top timestamps of keyframes matching a text query.

    Args:
        query: Text query describing what to search for (e.g., "person walking", "car driving")
        top_k: Number of top results to return (default: 3)
        video_id: Optional video ID to filter results to a specific video
        youtube_url: Optional YouTube URL to filter results. Takes precedence over video_id if both provided.

    Returns:
        Comma-separated string of timestamps in seconds (e.g., "5.0, 12.5, 18.0")
    """
    return _get_visual_timestamps_impl(query, top_k, video_id, youtube_url)


if __name__ == "__main__":
    """Example usage when run as a script."""
    import os
    import sys
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables from lively/.env
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)

    # Example query
    query = sys.argv[1] if len(sys.argv) > 1 else "person walking"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Searching for: '{query}'")
    print(f"Top K: {top_k}")
    print("-" * 80)

    try:
        timestamps_str = get_visual_timestamps(
            query=query,
            search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
            index_name=os.getenv("SEARCH_INDEX_NAME", "video-keyframes-index"),
            search_api_key=os.getenv("SEARCH_API_KEY"),
            top_k=top_k
        )

        print(f"\nResult: {timestamps_str}")
        print(f"\nTimestamps as list: {timestamps_str.split(', ')}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
