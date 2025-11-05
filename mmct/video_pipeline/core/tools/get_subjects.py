from mmct.providers.factory import provider_factory
from typing import Annotated, List, Dict, Any, Optional
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


async def get_subjects(
    query: Annotated[str, "query to search for subjects"],
    index_name: Annotated[str, "name of the search index"],
    video_id: Annotated[Optional[str], "unique identifier for the video"] = None,
    url: Annotated[Optional[str], "url of the video"] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve all subjects from the subject registry index filtered by video_id or url.

    Args:
        query: Search query string
        index_name: Name of the search index
        video_id: Optional unique identifier for the video
        url: Optional URL of the video

    Returns:
        List of subject documents matching the given filters.
    """
    # Construct the full index name
    full_index_name = f"subject-registry-{index_name}"

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
        #create embedding of query and pass
        # Search for all subjects matching the filter
        results = await search_provider.search(
            query=query,
            index_name=full_index_name,
            search_text="*",
            filter=filter_query,
            query_type="semantic",
            select=["subject_registry", "subject_count", "video_summary","video_id"],
        )
        return list(results)

    except Exception as e:
        print(f"Error fetching subjects for video_id={video_id} or url={url}: {e}")
        return []
    finally:
        await search_provider.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        index_name = "test"
        video_id = "d678544d517a57050f6a6881b0eb26496536053c45711ac624104cd2fccc00dc"

        print(f"Fetching subjects for video_id: {video_id}")
        subjects = await get_subjects(
            query="sample query",
            index_name=index_name,
            video_id=video_id
        )
        print(subjects)

    asyncio.run(main())