"""
This is a retreive documents tool which provide the summary with the transcript of video related to the query.
"""

# Importing Libraries
import os
from typing_extensions import Annotated, Optional
from mmct.providers.factory import provider_factory
from loguru import logger

try:
    logger.info("Instantiating the embedding and search providers")
    search_provider = provider_factory.create_search_provider()
    embed_provider = provider_factory.create_embedding_provider()
    logger.info("Successfully instantiated the search and embedding providers")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating providers: {e}")


async def get_context(
    query: Annotated[str, "query for which chapter documents need to be fetched."],
    index_name: Annotated[str, "vector index name"],
    video_id: Annotated[str, "video id if provided in the instruction"]=None,
    url: Annotated[str, "url if provided in the instruction to filter out the search results"] = None,
    start_time: Annotated[Optional[float], "start time in seconds to filter documents with overlapping time range"] = None,
    end_time: Annotated[Optional[float], "end time in seconds to filter documents with overlapping time range"] = None,
    fields_to_retrieve: Annotated[Optional[list], "list of fields to retrieve from the chapter index"] = None,
    top: Annotated[Optional[int], "number of top results to retrieve"] = 3,
) -> str:
    """
    Description:
        Retrieves relevant video chapters containing transcript chunk, visual summaries, actions,
        and text from scenes.

    Input Parameters:
        - query (str): query for which chapter documents need to be fetched.
        - index_name (str): Vector index name for search
        - video_id (Optional[str]): Video identifier (use from get_video_summary or get_object_collection if available)
        - url (Optional[str]): Video URL (alternative to video_id)
        - start_time (Optional[float]): Start time in seconds to filter documents (returns docs with overlapping time range)
        - end_time (Optional[float]): End time in seconds to filter documents (returns docs with overlapping time range)
        - fields_to_retrieve - Available fields:
            * chapter_transcript: Transcript with timestamps for this segment
            * detailed_summary: Visual summary of what happens in the chapter
            * action_taken: Specific actions performed or demonstrated
            * text_from_scene: Text visible in video (signs, captions, etc.)
            * object_collection: JSON string of objects in this chapter
            * start_time: Chapter start time in seconds
            * end_time: Chapter end time in seconds
            * hash_video_id: Video identifier
            * url: Video URL
        - top: Number of top results to retrieve

    Output:
        List of chapter documents, each containing fields based on fields_to_retrieve:
        - chapter_transcript (str): Transcript with timestamps for this segment
        - detailed_summary (str): Visual summary of what happens in the chapter
        - action_taken (str): Specific actions performed or demonstrated
        - text_from_scene (str): Text visible in video (signs, captions, etc.)
        - object_collection (str): JSON string of objects in this chapter
        - start_time (float): Chapter start time in seconds
        - end_time (float): Chapter end time in seconds
        - hash_video_id (str): Video identifier
        - url (str): Video URL
    """
    global search_provider, embed_provider

    

    # embedding the query
    embedding = await embed_provider.embedding(query)

    # Build filter query with multiple conditions
    filter_conditions = dict()
    
    if url:
        filter_conditions["url"] = {"eq": url}
    if video_id:
        if len(video_id)==64:
            parent_id = video_id
        else:
            parent_id = video_id[:64]
        filter_conditions["parent_id"] = {"eq": parent_id}
    
    # Add time overlap filter if both start_time and end_time are provided
    # Overlap condition: doc.start_time < end_time AND doc.end_time > start_time
    if start_time is not None and end_time is not None:
        filter_conditions["start_time"] = {"le": end_time}
        filter_conditions["end_time"] = {"ge": start_time}
    
    # Combine all filter conditions with 'and'
    if not filter_conditions:
        filter_conditions = None

    search_results = await search_provider.search(
        query=query,
        index_name=index_name,
        search_text=None,
        query_type="semantic",
        top=top,
        filter=filter_conditions,
        select=fields_to_retrieve,
        embedding=embedding
    )
    return search_results


if __name__ == "__main__":
    import asyncio

    video_id = "hash-video-id"
    query = "user-query"
    index_name = "index-name"
    url = "video-url"

    start_time = "start time in seconds"
    end_time = "end time in seconds"

    fields_to_retrieve = ["chapter_transcript","detailed_summary","action_taken"]

    
    results = asyncio.run(get_context(
        video_id=video_id, 
        query=query,
        index_name=index_name,
        start_time=start_time,
        end_time=end_time,
        url=url
    ))
    print(results)