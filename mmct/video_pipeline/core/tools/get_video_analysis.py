from mmct.providers.factory import provider_factory
from typing import Annotated, List, Dict, Any, Optional
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize search provider
search_provider = provider_factory.create_search_provider()
embed_provider = provider_factory.create_embedding_provider()

async def get_video_analysis(
    query: Annotated[str, "query to search for video analysis including objects, summary, and context"],
    index_name: Annotated[str, "name of the search index"],
    video_id: Annotated[Optional[str], "unique identifier for the video"] = None,
    url: Annotated[Optional[str], "url of the video"] = None,
    top: Annotated[Optional[int], "number of top results to retrieve"] = 3,
    fields_to_retrieve: Annotated[Optional[List[str]], "list of fields to retrieve from the object collection index"] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve high-level video analysis including object collection and video summary.

    Description:
        Retrieves comprehensive video analysis from the object collection index, including video summary,
        object collection with detailed attributes, and metadata.

    Input Parameters:
        - query (str): Search query to find relevant video analysis
        - index_name (str): Name of the search index
        - video_id (Optional[str]): Unique identifier for the video (use if available)
        - url (Optional[str]): URL of the video (alternative to video_id)
        - fields_to_retrieve: Available fields:
            * video_summary: Overall narrative and scene context of the video
            * object_collection: JSON string containing list of objects with:
                - name: Object identifier (e.g., "Person in blue shirt", "Red car")
                - appearance: List of visual characteristics
                - identity: List of type, category, role information
                - first_seen: Timestamp in seconds when object first appears
                - additional_details: Extra contextual information
            * object_count: Total number of unique objects identified
            * video_id: Video identifier
        - top: Number of top results to retrieve


    Output:
        List of dictionaries containing request fields
    """

    # Construct the full index name
    full_index_name = f"object-collection-{index_name}"

    try:
        # Build filter query
        filter_query = None
        if url:
            filter_query = f"url eq '{url}'"
        elif video_id:
            filter_query = f"video_id eq '{video_id}'"
        
        # embedding generation for the query
        embedding = await embed_provider.embedding(query)

        # Search for relevant video analysis matching the filter
        results = await search_provider.search(
            query=query,
            index_name=full_index_name,
            search_text="*",
            filter=filter_query,
            embedding=embedding,
            query_type="semantic",
            top=top,
            select=fields_to_retrieve,
            embedding_field_name="video_summary_embedding"
        )
        return list(results)

    except Exception as e:
        print(f"Error fetching video analysis for video_id={video_id} or url={url}: {e}")
        return []
    finally:
        await search_provider.close()
        

if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        index_name = "<index-name>"
        video_id = "<video-id>" # if available

        # print(f"Fetching video analysis for video_id: {video_id}")
        analysis = await get_video_analysis(
            query="<your-query-here>",
            index_name=index_name,
            video_id=video_id
        )
        print(analysis)

    asyncio.run(main())