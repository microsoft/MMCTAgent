"""
This tool provides the granular visual information related to query
"""
from loguru import logger
import os
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from typing_extensions import Annotated

from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher



def _extract_video_id(url: str) -> Optional[str]:
    """Derive the video_id from supported YouTube URL shapes."""
    if not url:
        return None

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        query_params = parse_qs(parsed.query or "")
        video_values = query_params.get("v")
        return video_values[0] if video_values else None

    if host == "youtu.be":
        return parsed.path.lstrip("/") or None

    return None

async def get_relevant_frames(
    query: Annotated[str, 'query to be look for frames'],
    url: Annotated[str, 'YouTube video URL'],
    index_name: Annotated[str, 'search index name'],
    top_k: Annotated[int, 'number of relevant frames to fetch'] = 5,
    start_time: Annotated[Optional[float], 'Optional start time (seconds) to constrain frame search'] = None,
    end_time: Annotated[Optional[float], 'Optional end time (seconds) to constrain frame search'] = None,
) -> List[dict]:
    """
    Discover relevant frame timestamps based on visual queries when timestamps are unknown.

    Description:
        Searches keyframe index to find relevant frames based on visual embeddings.
        Returns timestamps that can be passed to downstream timestamp-aware tools.

    Input Parameters:
        - query (str): [Mandatory] Visual description of what to search for (e.g., "frames showing person exercising")
        - url (str): [Mandatory] YouTube URL containing the video ID
        - index_name (str): [Mandatory] Search index name for keyframe search
        - top_k (int): [Mandatory] Number of relevant frames to retrieve (default: 5)
        - start_time (float, optional): Minimum timestamp (inclusive) for results
        - end_time (float, optional): Maximum timestamp (inclusive) for results

    Output:
        List of dicts with keys:
            - "timestamp": float timestamp (seconds)
            - "blob_url": blob URL for the frame, if available

    Workflow:
        1. Searches keyframe index using visual embeddings
        2. Returns frame timestamps as a list
    """
    try:
        video_id = _extract_video_id(url)
        if not video_id:
            return []
        
        # print(f"Extracted video_id: {video_id} from URL: {url}")
        # Get search endpoint from environment
        search_endpoint = os.getenv('SEARCH_ENDPOINT')
        # If there is a FAISS index directory in examples/ (e.g. from exported indices), prefer it
        provider_config = None
        alt_faiss_dir = os.path.join(os.getcwd(), "examples", "mmct_faiss_indices")
        default_faiss_dir = os.path.join(os.getcwd(), "mmct_faiss_indices")
        if os.path.isdir(alt_faiss_dir) and any(os.scandir(alt_faiss_dir)):
            provider_config = {"index_path": alt_faiss_dir}
        elif os.path.isdir(default_faiss_dir) and any(os.scandir(default_faiss_dir)):
            provider_config = {"index_path": default_faiss_dir}
        
        searcher = KeyframeSearcher(
            search_endpoint=search_endpoint,
            index_name=f"keyframes-{index_name}",
            provider_config=provider_config,
        )
        
        filters = [f"video_id eq '{video_id}'"]
        if start_time is not None:
            filters.append(f"timestamp_seconds ge {start_time}")
        if end_time is not None:
            filters.append(f"timestamp_seconds le {end_time}")
        video_filter = " and ".join(filters)
        # Search for relevant frames
        results = await searcher.search_keyframes(
            query=query,
            top_k=top_k,
            video_filter=video_filter
        )
        
        if not results:
            return []


        # Extract timestamps and blob URLs from results metadata
        frame_matches: List[dict] = []
        # Results may come from different providers with different shapes:
        # - Azure: result is a document dict with fields at top-level
        # - Local FAISS: result is {'id', 'score', 'document': { ... }}
        for result in results:
            # normalize to a document dict
            if isinstance(result, dict) and 'document' in result and isinstance(result['document'], dict):
                doc = result['document']
            elif isinstance(result, dict):
                doc = result
            else:
                # unexpected shape
                continue

            # apply video_id filter locally in case provider didn't support it
            doc_video_id = doc.get('video_id')
            if doc_video_id is not None and doc_video_id != video_id:
                continue

            blob_url = doc.get('blob_url')
            timestamp_raw = doc.get('timestamp_seconds')
            if timestamp_raw is None:
                continue

            try:
                timestamp_value = float(timestamp_raw)
            except (TypeError, ValueError):
                continue

            if start_time is not None and timestamp_value < start_time:
                continue
            if end_time is not None and timestamp_value > end_time:
                continue

            frame_matches.append({
                "timestamp": timestamp_value,
                "blob_url": blob_url
            })


        return frame_matches
        
    except Exception as e:
        return []


if __name__ == "__main__":
    import asyncio
    
    async def main():
        res = await get_relevant_frames(
            query="Derivative of Gaussian graph",
            url="https://www.youtube.com/watch?v=9r8ph2pb9aw",
            index_name="kv-nptel-longer-chapter",
            top_k=3,
            start_time=0.0,
            end_time=60.0,
        )

        print("-------------- get_relevant_frames result:", res)
    
    asyncio.run(main())