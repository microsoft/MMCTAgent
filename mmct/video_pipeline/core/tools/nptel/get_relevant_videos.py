from mmct.providers.factory import provider_factory
from typing import Annotated, List, Dict, Any, Optional
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


FIELDS_TO_RETRIEVE: List[str] = [
    "video_summary",
    "video_id",
    "url",
    "video_duration",
]


def _format_video_results(results: List[Dict[str, Any]]) -> str:
    """Convert provider search results into a readable summary string."""
    if not results:
        return "No relevant videos found."

    formatted_blocks: List[str] = []
    for idx, raw_result in enumerate(results, start=1):
        score = None
        doc: Dict[str, Any] = {}
        if isinstance(raw_result, dict):
            score = raw_result.get("@search.score")
            if score is None:
                score = raw_result.get("score")
            if isinstance(raw_result.get("document"), dict):
                doc = raw_result["document"]
            else:
                doc = raw_result

        video_id = doc.get("video_id") or "unknown"
        url = doc.get("url") or "unknown"
        summary = doc.get("video_summary") or "Summary unavailable."
        duration = doc.get("video_duration")
        duration_text = f"Duration: {duration}s" if duration is not None else "Duration: unknown"
        score_text = f"score={score:.4f}" if isinstance(score, (int, float)) else "score=unknown"

        block = (
            f"Video {idx} ({score_text})\n"
            f"Video ID: {video_id} | URL: {url}\n"
            f"{duration_text}\n"
            f"Summary: {summary}"
        )
        formatted_blocks.append(block)

    return "\n\n".join(formatted_blocks)


async def get_relevant_videos(
    query: Annotated[str, "query to search for related video summaries"],
    index_name: Annotated[str, "name of the search index provided in the user query"],
    video_id: Annotated[Optional[str], "unique identifier for the video aka hash Id"] = None,
    url: Annotated[Optional[str], "url of the video"] = None,
    top: Annotated[Optional[int], "number of top results to retrieve (max 3)"] = 3
) -> str:
    """
    Description:
        Retrieve high-level video summaries of relevant videos.

        This tool is used for:
        1. Video discovery: Call WITHOUT video_id/URL to find relevant videos matching the query
        2. Specific video summary: Call WITH video_id/URL to get summary of a specific video

        IMPORTANT: If video_id or URL is not provided in the instruction, always call this tool
        first to discover relevant videos and obtain their video_ids for subsequent tool calls.

    Input Parameters:
        - query (str): query to search for related video summaries, this is mandatory field
        - index_name (str): Name of the search index
        - video_id (Optional[str]): Unique identifier for the video (use if available, otherwise omit)
        - url (Optional[str]): URL of the video (use if available, otherwise omit)
        - top: Number of top results to retrieve


    Output:
        Formatted string enumerating the most relevant videos, including similarity score.
    """

    # Construct the full index name
    full_index_name = f"object-collection-{index_name}"

    # Get search endpoint from environment
    search_endpoint = os.getenv("SEARCH_ENDPOINT")
    if not search_endpoint:
        raise ValueError("SEARCH_ENDPOINT environment variable not set")

    # Initialize search provider
    search_provider = provider_factory.create_search_provider()

    # Initalize Embedding provider
    embed_provider = provider_factory.create_embedding_provider()

    # embedding the query
    embedding = await embed_provider.embedding(query)

    try:
        # Build filter query
        filter_query = None
        if url:
            filter_query = f"url eq '{url}'"
        elif video_id:
            filter_query = f"video_id eq '{video_id}'"

        # Search for video summary matching the filter
        results = await search_provider.search(
            query=query,
            index_name=full_index_name,
            search_text=None,
            filter=filter_query,
            query_type="semantic",
            top=top,
            select=FIELDS_TO_RETRIEVE,
            embedding=embedding
        )

        return _format_video_results(list(results))

    except Exception as e:
        print(f"Error fetching video summary for video_id={video_id} or url={url}: {e}")
        return "Failed to retrieve video summaries."
    finally:
        # await search_provider.close()
        pass


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        index_name = "kv-nptel-longer-chapter"
        video_id = "<hash-video-id>"
        query = "How and why are images represented as 2D matrices?"

        print(f"Fetching video summary for video_id: {video_id}")
        summary = await get_relevant_videos(
            query=query,
            index_name=index_name,
            # video_id=video_id
        )
        print(summary)

    asyncio.run(main())
