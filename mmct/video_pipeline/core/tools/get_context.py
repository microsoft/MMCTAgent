"""
This is a retreive documents tool which provide the summary with the transcript of video related to the query.
"""

# Importing Libraries
import os
from typing_extensions import Annotated
from mmct.video_pipeline.utils.helper import get_media_folder
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory
from loguru import logger
from mmct.config.settings import SearchConfig

try:
    config = MMCTConfig()
    logger.info("Successfully retrieved the MMCT config")
except Exception as e:
    logger.exception(f"Exception occurred while fetching the MMCT config: {e}")

try:
    logger.info("Instantiating the embedding and search providers")
    search_provider = provider_factory.create_search_provider(
        config.search.provider, config.search.model_dump()
    )
    embed_provider = provider_factory.create_embedding_provider(
        provider_name=config.embedding.provider, config=config.embedding.model_dump()
    )
    logger.info("Successfully instantiated the search and embedding providers")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating providers: {e}")


async def get_context(
    query: Annotated[str, "query for which documents needs to fetch"],
    index_name: Annotated[str, "vector index name"],
    video_id: Annotated[str, "video id if provided in the instruction"] = None,
    url: Annotated[str, "url if provided in the instruction to filter out the search results"] = None,
    use_graph_rag: Annotated[str, "whether to use graph rag or not"] = 'False',
) -> str:
    """
    retreive related documents based on the query from the vector database.
    """
    global search_provider, embed_provider
    # embedding the query
    embedding = await embed_provider.embedding(query)
   
    
    if use_graph_rag=='True':
        search_config = SearchConfig(provider="custom_search")
        search_provider = provider_factory.create_search_provider(
            search_config.provider, search_config.model_dump()
        )
        search_results = await search_provider.search(query=query, index_name="temp",embedding=embedding)
        return search_results

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
        ],
        embedding=embedding
    )
    return search_results


if __name__ == "__main__":
    import asyncio

    video_id = "hash-video-id"
    query = "user-query"
    index_name = "index-name"
    results = asyncio.run(get_context(video_id=video_id, query=query,index_name= index_name,use_graph_rag=True))
    print(results)
