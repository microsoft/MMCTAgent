from mcp_server.tools.schemas.kb_tool_schemas import SearchRequest, get_filter_string
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from typing import Annotated, List, Dict
from mcp_server.server import mcp
from loguru import logger

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


@mcp.tool(
    name="kb_tool",
    description="""The Knowledge Base Search Tool (kb_tool) allows agents to retrieve structured metadata from an AI Search index based on a user's query. 
    
It supports three search modes:

1. full → keyword-based full-text search
2. vector → embedding similarity search
3. semantic → semantic ranking with natural language understanding

Agents can apply filters (category, sub-category, subject, variety, time range, or video ID) to narrow results, and use the select parameter to return only specific fields from the indexed documents (e.g., category, sub_category, subject, variety, hash_video_id).

The tool returns the top-k most relevant results along with optional semantic answers, enabling agents to extract meaningful information from a video knowledge base or similar structured datasets.

## Input Schema
- query (string, required) → The search text or * for full index scans (only valid with full search).
- query_type (string, default=full) → One of full, vector, or semantic.
- index_name (string, required) → Target Azure AI Search index name.
- k (integer, default=10) → Number of top results to return.
- filters (object, optional) → Filtering options:
- category (string)
- sub_category (string)
- subject (string)
- variety (string)
- time_from / time_to (ISO datetime strings)
- hash_video_id (string)
- select (list of strings, optional) → Specific fields to include in results.

## Output

A list of result dictionaries, each containing metadata fields retrieved from the search index, based on the query, mode, and filters applied.""",
)
async def kb_tool(
    request: Annotated[
        SearchRequest, "Search parameters including query, mode, k, and optional filters"
    ],
) -> List[Dict]:
    """
    Perform search over video chapter knowledge base, using chosen mode and filters,
    returning top-k results with optional semantic answer.
    """
    embedding = None

    if (request.query and request.query in ["*"]) and (
        request.query_type and request.query_type in ["vector", "semantic"]
    ):
        raise Exception("Invalid input segment. For * queries, query type must be `full`")

    if request.query_type in ("vector", "semantic"):
        embedding = await embed_provider.embedding(text=request.query)

    results = await search_provider.search(
        index_name=request.index_name,
        query=request.query,
        query_type=request.query_type,
        top=request.k,
        filter=await get_filter_string(request.filters.model_dump()) if request.filters else None,
        embedding=embedding,
        select=request.select if request.select else None
    )

    return results