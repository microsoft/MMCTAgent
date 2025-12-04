from typing import Annotated, List, Dict, Any, Optional
from mmct.providers.base import BaseObjectCollectionVectorDBProvider, BaseEmbeddingProvider
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class GetVideoSummaryTool:
    def __init__(self, vectordb_object_registry:BaseObjectCollectionVectorDBProvider, embed_provider:BaseEmbeddingProvider):
        self.vectordb_object_registry = vectordb_object_registry
        self.embed_provider = embed_provider
        
    async def get_video_summary(
        self,
        query: Annotated[str, "query to search for related video summaries"],
        index_name: Annotated[str, "name of the search index provided in the user query"],
        video_id: Annotated[Optional[str], "unique identifier for the video aka hash Id"] = None,
        url: Annotated[Optional[str], "url of the video"] = None,
        top: Annotated[Optional[int], "number of top results to retrieve (max 3)"] = 3
    ) -> List[Dict[str, Any]]:
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
            List of dictionaries containing requested fields, including video_id for use in other tools
        """

        # embedding the query
        embedding = await self.embed_provider.embedding(query)

        try:
            # Build filter conditions
            filter_conditions = dict()
            if url:
                filter_conditions['url'] = {'eq': url}
            elif video_id:
                filter_conditions['video_id'] = {'eq': video_id}

            # Search for video summary matching the filter
            results = await self.vectordb_object_registry.search(
                query=query,
                search_text=None,
                filter=filter_conditions,
                query_type="semantic",
                top=top,
                embedding=embedding
            )

            fields_to_retrieve=['video_summary','video_id','url']
            results_with_scores = []
            for document, score in results:
                doc_dict = document.model_dump()

                # Only include fields specified by the user
                filtered_dict = {
                    field: doc_dict.get(field) for field in fields_to_retrieve
                    if field in doc_dict
                }
                filtered_dict['@search.score'] = score
                results_with_scores.append(filtered_dict)

            return results_with_scores

        except Exception as e:
            print(f"Error fetching video summary for video_id={video_id} or url={url}: {e}")
            return []
        finally:
            # await search_provider.close()
            pass


if __name__ == "__main__":
    import asyncio

    async def main():
        # Example usage
        index_name = "<index-name>"
        video_id = "<hash-video-id>"
        query = "<sample-query>"
        get_video_summary_tool_object = GetVideoSummaryTool()
        print(f"Fetching video summary for video_id: {video_id}")
        summary = await get_video_summary_tool_object.get_video_summary(
            query=query,
            index_name=index_name,
            video_id=video_id
        )
        print(summary)

    asyncio.run(main())