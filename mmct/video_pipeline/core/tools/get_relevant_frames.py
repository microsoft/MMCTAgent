"""
This tool provides the granular visual information related to query
"""

import os
import shutil
from typing_extensions import Annotated
from typing import List, Dict, Any, Optional
from mmct.video_pipeline.core.tools.utils.search_keyframes import KeyframeSearcher
from mmct.providers.base import BaseKeyframesVectorDBProvider, BaseImageEmbeddingProvider


class GetRelevantFrames:
    def __init__(
        self,
        vectordb_keyframes: BaseKeyframesVectorDBProvider,
        image_embedding_provider: BaseImageEmbeddingProvider,
    ):
        self.vectordb_keyframes = vectordb_keyframes
        self.image_embedding_provider = image_embedding_provider  # For CLIP embeddings

    async def get_relevant_frames(
        self,
        query: Annotated[str, "query to be look for frames"],
        video_id: Annotated[str, "video id"],
        top_k: Annotated[int, "number of relevant frames to fetch"] = 5,
    ) -> List[str]:
        """
        Discover relevant frame IDs based on visual queries when specific timestamps are unknown. 
        
        This tool is a 'last resort' for the planner when object tracking or context summarized times are insufficient.
        Returns a list of frame objects (path and timestamp) that can be passed to 'query_frame' for visual analysis.
        """
        try:
            # If there is a FAISS index directory in examples/ (e.g. from exported indices), prefer it
            provider_config = None
            alt_faiss_dir = os.path.join(os.getcwd(), "examples", "mmct_faiss_indices")
            default_faiss_dir = os.path.join(os.getcwd(), "mmct_faiss_indices")
            if os.path.isdir(alt_faiss_dir) and any(os.scandir(alt_faiss_dir)):
                provider_config = {"index_path": alt_faiss_dir}
            elif os.path.isdir(default_faiss_dir) and any(os.scandir(default_faiss_dir)):
                provider_config = {"index_path": default_faiss_dir}

            searcher = KeyframeSearcher(
                search_provider=self.vectordb_keyframes,
                image_embedding_provider=self.image_embedding_provider,
                provider_config=provider_config,
            )

            video_filter = dict()
            if video_id:
                video_filter["video_id"] = {"eq": video_id}

            # Search for relevant frames
            results = await searcher.search_keyframes(
                query=query, top_k=top_k, video_filter=video_filter if video_filter else None
            )

            if not results:
                return []

            # Extract keyframe filenames from results
            response = []
            # Results may come from different providers with different shapes:
            # - Azure: result is a document dict with fields at top-level
            # - Local FAISS: result is {'id', 'score', 'document': { ... }}
            for result in results:
                # normalize to a document dict
                if (
                    isinstance(result, dict)
                    and "document" in result
                    and isinstance(result["document"], dict)
                ):
                    doc = result["document"]
                elif isinstance(result, dict):
                    doc = result
                else:
                    # unexpected shape
                    continue

                # apply video_id filter locally in case provider didn't support it
                doc_video_id = doc.get("video_id")
                if doc_video_id is not None and doc_video_id != video_id:
                    continue

                keyframe_filename = doc.get("keyframe_filename") or ""
                timestamp_seconds = doc.get("timestamp_seconds")
                if keyframe_filename:
                    # keep the filename (right-most segment) for downstream consumers
                    response.append(
                        {
                            "frame_path": f"{keyframe_filename}",
                            "timestamps": timestamp_seconds,
                        }
                    )

            return response

        except Exception as e:
            return []


if __name__ == "__main__":
    import asyncio

    async def main():
        get_relevant_frame_object = GetRelevantFrames()
        res = await get_relevant_frame_object.get_relevant_frames(
            query="user-query", video_id="hash-video-id",top_k=10
        )

        print("get_relevant_frames result:", res)

    asyncio.run(main())
