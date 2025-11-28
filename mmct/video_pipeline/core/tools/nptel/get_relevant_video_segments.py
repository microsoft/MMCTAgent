"""
Retrieve and summarize relevant video chapters for a given query.
"""

# Importing Libraries
from typing import List
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


FIELDS_TO_RETRIEVE: List[str] = [
    "chapter_transcript",
    "detailed_summary",
    "start_time",
    "end_time",
    "hash_video_id",
]


def _format_segment_results(results: List[dict]) -> str:
    """Convert search provider documents into a human-readable string."""
    if not results:
        return "No relevant video segments found."

    formatted_segments: List[str] = []
    for idx, raw_doc in enumerate(results, start=1):
        # Normalize shape: providers may wrap payload under 'document'
        score = None
        if isinstance(raw_doc, dict):
            score = raw_doc.get("@search.score")
            if score is None:
                score = raw_doc.get("score")
            doc = raw_doc.get("document")
        else:
            doc = None
        if not isinstance(doc, dict):
            doc = raw_doc if isinstance(raw_doc, dict) else {}

        hash_video_id = doc.get("hash_video_id") or "unknown"
        start_time = doc.get("start_time")
        end_time = doc.get("end_time")
        detailed_summary = doc.get("detailed_summary") or "Summary unavailable."
        chapter_transcript = doc.get("chapter_transcript") or "Transcript unavailable."

        time_window = "unknown"
        if start_time is not None and end_time is not None:
            time_window = f"{start_time:.2f}s - {end_time:.2f}s"
        elif start_time is not None:
            time_window = f"from {start_time:.2f}s"
        elif end_time is not None:
            time_window = f"until {end_time:.2f}s"

        score_text = f"score={score:.4f}" if isinstance(score, (int, float)) else "score=unknown"
        segment_text = (
            f"Segment {idx} ({score_text}) (Video: {hash_video_id}, Time: {time_window})\n"
            f"Summary(Includes visual info from video frames): {detailed_summary}\n"
            f"Transcript: {chapter_transcript}"
        )
        formatted_segments.append(segment_text)

    return "\n\n".join(formatted_segments)


async def get_relevant_video_segments(
    query: Annotated[str, "query for which chapter documents need to be fetched."],
    index_name: Annotated[str, "vector index name"],
    video_id: Annotated[str, "video id if provided in the instruction"]=None,
    url: Annotated[str, "url if provided in the instruction to filter out the search results"] = None,
    start_time: Annotated[Optional[float], "start time in seconds to filter documents with overlapping time range"] = None,
    end_time: Annotated[Optional[float], "end time in seconds to filter documents with overlapping time range"] = None,
    top: Annotated[Optional[int], "number of top results to retrieve"] = 3,
) -> str:
    """
    Description:
        Retrieves relevant video chapters containing transcript chunk, visual summaries, actions,
        and text from scenes.

    Input Parameters:
        - query (str): Query describing the desired content.
        - index_name (str): Vector index name for search.
        - video_id (Optional[str]): Video identifier (use from get_video_summary or get_object_collection if available).
        - url (Optional[str]): Video URL (alternative to video_id).
        - start_time (Optional[float]): Start time in seconds to filter documents (returns docs with overlapping time range).
        - end_time (Optional[float]): End time in seconds to filter documents (returns docs with overlapping time range).
        - top (int): Number of top results to retrieve.

    Output:
        Formatted string describing the most relevant segments (summary, transcript, and timing).
    """
    global search_provider, embed_provider
    # embedding the query
    embedding = await embed_provider.embedding(query)

    # Build filter query with multiple conditions
    filter_conditions = []
    
    if url:
        filter_conditions.append(f"youtube_url eq '{url}'")
    elif video_id:
        filter_conditions.append(f"hash_video_id eq '{video_id}'")
    
    # Add time overlap filter if both start_time and end_time are provided
    # Overlap condition: doc.start_time < end_time AND doc.end_time > start_time
    if start_time is not None and end_time is not None:
        filter_conditions.append(f"(start_time lt {end_time} and end_time gt {start_time})")
    
    # Combine all filter conditions with 'and'
    if filter_conditions:
        filter_query = " and ".join(filter_conditions)
    else:
        filter_query = None  # no filter

    search_results = await search_provider.search(
        query=query,
        index_name=index_name,
        search_text=None,
        query_type="vector",
        top=top,
        filter=filter_query,
        select=FIELDS_TO_RETRIEVE,
        embedding=embedding
    )
    return _format_segment_results(search_results)


if __name__ == "__main__":
    import asyncio

    video_id = "9r8ph2pb9aw"
    query = "How is the derivative equation of discrete data derived from continous function?"
    index_name = "kv-nptel-longer-chapter"
    # Example: fetch documents with time overlap between 10.0 and 30.0 seconds
    formatted_response = asyncio.run(get_relevant_video_segments(
        video_id=video_id,
        query=query,
        index_name=index_name,
        start_time=600,
        end_time=700
    ))
    print(formatted_response)