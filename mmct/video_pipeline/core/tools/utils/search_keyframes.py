#!/usr/bin/env python3
"""
Keyframe Search Script

This script searches for keyframes in the Azure AI Search index by:
1. Taking a text query from user
2. Generating CLIP embedding for the query
3. Finding top 5 most similar keyframes
4. Displaying results with details
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

from mmct.video_pipeline.core.tools.utils.azure_search_client import VideoFrameSearchClient
from mmct.video_pipeline.core.tools.utils.embeddings_generator import EmbeddingsGenerator, EmbeddingConfig


class KeyframeSearcher:
    """Search for keyframes using text queries."""

    def __init__(self,
                 search_endpoint: str,
                 search_key: Optional[str] = None,
                 index_name: str = "video-frames-index",
                 clip_model: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the keyframe searcher.

        Args:
            search_endpoint: Azure AI Search endpoint
            search_key: Azure AI Search API key (optional)
            index_name: Search index name
            clip_model: CLIP model name for query embeddings
        """
        # Initialize search client
        self.search_client = VideoFrameSearchClient(
            search_endpoint=search_endpoint,
            search_key=search_key,
            index_name=index_name
        )

        # Initialize embeddings generator for query encoding
        embeddings_config = EmbeddingConfig(
            clip_model_name=clip_model,
            batch_size=1
        )
        self.embeddings_generator = EmbeddingsGenerator(embeddings_config)


    async def search_keyframes(self,
                        query: str,
                        top_k: int = 5,
                        video_filter: Optional[str] = None,
                        min_motion_score: Optional[float] = None,
                        min_similarity_score: Optional[float] = 0.7) -> List[Dict[str, Any]]:
        """
        Search for keyframes using text query.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            video_filter: Optional filter for specific video name/path
            min_motion_score: Optional minimum motion score filter
            min_similarity_score: Minimum similarity score threshold (0.0-1.0). Default 0.7

        Returns:
            List of search results filtered by similarity score
        """
        try:

            # Generate query embedding
            query_embedding = await self.embeddings_generator.generate_text_embedding(query)

            # Build filter expression
            filters = f"{video_filter}" if video_filter else None

            # Search for similar frames
            results = await self.search_client.search_similar_frames(
                query_vector=query_embedding,
                query_text=query,
                top_k=top_k,
                filters=filters
            )

            # Filter by similarity score to remove noisy/irrelevant frames
            if min_similarity_score is not None and results:
                print(results)
                filtered_results = [
                    r for r in results
                    if r.get('@search.score', 0) >= min_similarity_score
                ]
                return filtered_results

            return results

        except Exception as e:
            return []

