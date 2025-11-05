"""
This is a retreive documents tool which provide the summary with the transcript of video related to the query.
"""

# Importing Libraries
import os
from typing_extensions import Annotated
from mmct.video_pipeline.utils.helper import get_media_folder
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
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
    query: Annotated[str, "query for which documents needs to fetch"],
    index_name: Annotated[str, "vector index name"],
    video_id: Annotated[str, "video id if provided in the instruction"] = None,
    url: Annotated[str, "url if provided in the instruction to filter out the search results"] = None,
) -> str:
    """
    retreive related documents based on the query from the vector database.
    """
    global search_provider, embed_provider
    # embedding the query
    embedding = await embed_provider.embedding(query)

    if url:
        filter_query = f"youtube_url eq '{url}'"
    elif video_id:
        filter_query = f"hash_video_id eq '{video_id}'"
    else:
        filter_query = None  # no filter

    search_results = await search_provider.search(
        query=query,
        index_name=index_name,
        search_text=None,
        query_type="vector",
        top=5,
        filter=filter_query,
        select=[
            "detailed_summary",
            "topic_of_video",
            "action_taken",
            "text_from_scene",
            "chapter_transcript",
            "hash_video_id",
            "youtube_url",
            "subject_registry",
            "end_time",
            "start_time",
        ],
        embedding=embedding
    )
    return search_results


if __name__ == "__main__":
    import asyncio

    video_id = "hash-video-id"
    query = "user-query"
    index_name = "index-name"
    results = asyncio.run(get_context(video_id=video_id, query=query,index_name= index_name))
    print(results)
