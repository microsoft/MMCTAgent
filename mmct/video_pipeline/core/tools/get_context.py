"""
This is a retreive documents tool which provide the summary with the transcript of video related to the query.
"""

# Importing Libraries
from typing_extensions import Annotated, Optional
from mmct.providers.base import BaseEmbeddingProvider, BaseChapterVectorDBProvider


class GetContextTool:
    def __init__(self, embed_provider:BaseEmbeddingProvider, vectordb_chapter:BaseChapterVectorDBProvider):
        self.embed_provider = embed_provider
        self.vectordb_chapter = vectordb_chapter

    async def get_context(
        self,
        query: Annotated[str, "query for which chapter documents need to be fetched."],
        fields_to_retrieve: Annotated[list, "list of fields to retrieve from the chapter index"],
        video_id: Annotated[str, "video id if provided in the instruction"]=None,
        url: Annotated[str, "url if provided in the instruction to filter out the search results"] = None,
        start_time: Annotated[Optional[float], "start time in seconds to filter documents with overlapping time range"] = None,
        end_time: Annotated[Optional[float], "end time in seconds to filter documents with overlapping time range"] = None,
        top: Annotated[Optional[int], "number of top results to retrieve"] = 3,
    ) -> str:
        """
        Retrieves relevant video chapters containing transcripts, summaries, and actions.

        Available fields to retrieve in 'fields_to_retrieve':
        - chapter_transcript: Dialogue with timestamps.
        - detailed_summary: Visual summary of the chapter.
        - action_taken: Specific actions demonstrated.
        - text_from_scene: OCR text visible in video.
        - start_time / end_time: Segment boundaries in seconds.
        - hash_video_id / url: Identifiers.
        """

        # embedding the query
        embedding = await self.embed_provider.embedding(query)

        # Build filter query with multiple conditions
        filter_conditions = dict()
        
        if url:
            filter_conditions["url"] = {"eq": url}
        if video_id:
            filter_conditions["hash_video_id"] = {"eq": video_id}
        
        # Add time overlap filter if both start_time and end_time are provided
        # Overlap condition: doc.start_time < end_time AND doc.end_time > start_time
        if start_time is not None and end_time is not None:
            filter_conditions["start_time"] = {"le": end_time}
            filter_conditions["end_time"] = {"ge": start_time}
        
        # Combine all filter conditions with 'and'
        if not filter_conditions:
            filter_conditions = None

        search_results = await self.vectordb_chapter.search(
            query=query,
            search_text=None,
            query_type="semantic",
            top=top,
            filter=filter_conditions,
            embedding=embedding
        )

        # Convert new return type List[Tuple[ChapterIndexDocument, float]] to list of dicts
        results_with_scores = []
        for document, score in search_results:
            doc_dict = document.model_dump()

            # Only include fields specified by the user
            filtered_dict = {
                field: doc_dict.get(field) for field in fields_to_retrieve
                if field in doc_dict
            }
            filtered_dict['@search.score'] = score
            results_with_scores.append(filtered_dict)

        return results_with_scores


if __name__ == "__main__":
    import asyncio

    video_id = "hash-video-id"
    query = "user-query"
    url = "video-url"

    start_time = "start time in seconds"
    end_time = "end time in seconds"

    fields_to_retrieve = ["chapter_transcript","detailed_summary","action_taken"]

    context_tool_object = GetContextTool()
    results = asyncio.run(context_tool_object.get_context(
        video_id=video_id, 
        query=query,
        start_time=start_time,
        end_time=end_time,
        url=url
    ))
    print(results)