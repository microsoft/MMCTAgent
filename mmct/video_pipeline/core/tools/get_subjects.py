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
            top=1000,  # Get all subjects for the video
            select = ['name','appearance','identity','additional_details','first_seen']
        )
        results = [[res['name'], res['appearance'], res['identity'], res['additional_details'],res['first_seen']] for res in results]
        
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
        video_id = "d678544d517a57050f6a6881b0eb26496536053c45711ac624104cd2fccc00dc"
        index_name = "test"  # Replace with actual index name

        print(f"Fetching subjects for video_id: {video_id}")
        subjects = await get_subjects(video_id, index_name)

        print(subjects)

    asyncio.run(main())