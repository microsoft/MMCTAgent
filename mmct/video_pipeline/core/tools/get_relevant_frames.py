"""
This tool provides the granular visual information related to query
"""
import os
import shutil
from typing_extensions import Annotated
from typing import List, Dict, Any
from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher



async def get_relevant_frames(
    query: Annotated[str, 'query to be look for frames'], 
    video_id: Annotated[str, 'video id'],
    index_name: Annotated[str, 'search index name'],
    top_k: Annotated[int, 'number of relevant frames to fetch'] = 10,
) -> List[str]:
    """
    Fetch relevant frames for a query and return keyframe filenames.
    
    Args:
        query: Text query to search for
        video_id: Hash-based video ID to filter by
        top_k: Number of relevant frames to fetch
        index_name: Search index name
        
    Returns:
        List of keyframe filenames
    """
    try:
        
        # Get search endpoint from environment
        search_endpoint = os.getenv('SEARCH_ENDPOINT')
        
        # Initialize searcher
        searcher = KeyframeSearcher(
            search_endpoint=search_endpoint,
            index_name=f"keyframes-{index_name}",
        )
        
        video_filter = f"video_id eq '{video_id}'"
        # Search for relevant frames
        results = await searcher.search_keyframes(
            query=query,
            top_k=top_k,
            video_filter=video_filter
        )
        
        if not results:
            return []
        
        
        # Extract keyframe filenames from results
        keyframe_ids = []
        for result in results:
            keyframe_filename = result.get('keyframe_filename', '')
            if keyframe_filename:
                keyframe_ids.append(keyframe_filename.split("_")[-1])  # Extract only the filename part
        
        
        return keyframe_ids
        
    except Exception as e:
        return []


if __name__ == "__main__":
    import asyncio
    
    async def main():
        query = "test"
        video_id = "test"
        
        result = await get_relevant_frames(query, video_id, top_k=10)
        pass
    
    asyncio.run(main())