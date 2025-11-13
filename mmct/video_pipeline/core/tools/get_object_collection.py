from mmct.providers.factory import provider_factory
from typing import Annotated, List, Dict, Any, Optional
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


async def get_object_collection(
    index_name: Annotated[str, "name of the search index provided in the user query"],
    video_id: Annotated[Optional[str], "unique identifier for the video"] = None,
    url: Annotated[Optional[str], "url of the video"] = None,
) -> List[Dict[str, Any]]:
    """
    Description:
        Retrieve object collection data including object descriptions, counts, and first_seen timestamps.

        This tool is used for object tracking and object-related queries.

        IMPORTANT: This tool REQUIRES a valid video_id or URL before calling.
        - If video_id/URL is not provided in the instruction, call get_video_summary first to obtain it.
        - Use this tool for: object counting, object tracking, object appearance details, first_seen timestamps.

    Input Parameters:
        - index_name (str): Name of the search index
        - video_id (str): REQUIRED - Unique identifier for the video (obtain from get_video_summary if not provided)
        - url (str): REQUIRED if video_id not available - URL of the video

    Output:
        List of dictionaries containing requested fields
    """
    # Construct the full index name
    full_index_name = f"object-collection-{index_name}"

    # Get search endpoint from environment
    search_endpoint = os.getenv("SEARCH_ENDPOINT")
    if not search_endpoint:
        raise ValueError("SEARCH_ENDPOINT environment variable not set")

    # Initialize search provider
    search_provider = provider_factory.create_search_provider()
    
    try:
        # Build filter query
        filter_query = None
        if url:
            filter_query = f"url eq '{url}'"
        elif video_id:
            filter_query = f"video_id eq '{video_id}'"

        # Search for object collection matching the filter
        results = await search_provider.search(
            query = "*",
            index_name = full_index_name,
            search_text = "*",
            filter = filter_query,
            top = 1,
            select = ['object_collection','object_count','video_id'],
        )
        return list(results)

    except Exception as e:
        print(f"Error fetching object collection for video_id={video_id} or url={url}: {e}")
        return []
    finally:
        await search_provider.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        index_name = "<index-name>"
        video_id = "<hash-video-id>"
        query = "<sample-query>"

        print(f"Fetching object collection for video_id: {video_id}")
        collection = await get_object_collection(
            query=query,
            index_name=index_name,
            video_id=video_id
        )
        print(collection)

    asyncio.run(main())
