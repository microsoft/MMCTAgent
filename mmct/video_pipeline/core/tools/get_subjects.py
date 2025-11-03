from mmct.providers.factory import provider_factory
from typing import Annotated, List, Dict, Any
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


async def get_subjects(
    video_id: Annotated[str, "unique identifier for the video"],
    index_name: Annotated[str, "name of the search index"]
) -> List[Dict[str, Any]]:
    """
    Retrieve all subjects from the subject registry index filtered by video_id.

    Args:
        video_id: Unique identifier for the video
        index_name: Name of the search index

    Returns:
        List of subject documents matching the video_id
    """
    # Construct the full index name
    full_index_name = f"subject-registry-{index_name}"

    # Get search endpoint from environment
    search_endpoint = os.getenv('SEARCH_ENDPOINT')
    if not search_endpoint:
        raise ValueError("SEARCH_ENDPOINT environment variable not set")

    # Initialize search provider
    search_provider = provider_factory.create_search_provider()

    try:
        # Build filter for video_id
        video_filter = f"video_id eq '{video_id}'"

        # Search for all subjects matching the video_id
        results = await search_provider.search(
            query="*",
            index_name=full_index_name,
            search_text="*",
            filter=video_filter,
            select = ['subject_registry','subject_count']
        )
        results = [res['subject_registry'] for res in results]
        
        return results

    except Exception as e:
        print(f"Error fetching subjects for video_id {video_id}: {e}")
        return []
    finally:
        # Close the search provider
        await search_provider.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        video_id = "808ef24205b8bfe7181818699675f5a4dbfe5974baf5ded99ab5b5b3c8b6f15d"
        index_name = "bmark-video-mme-incorrect-responses-videos"  # Replace with actual index name

        print(f"Fetching subjects for video_id: {video_id}")
        subjects = await get_subjects(video_id, index_name)

        print(subjects)

    asyncio.run(main())