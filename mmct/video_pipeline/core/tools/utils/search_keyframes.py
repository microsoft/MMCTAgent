#!/usr/bin/env python3
"""
Keyframe Search Script

This script searches for keyframes using injected providers:
1. Taking a text query from user
2. Generating image embedding for the query using image_embedding_provider
3. Finding top k most similar keyframes using search_provider
4. Returning results
"""

from typing import List, Dict, Any, Optional
import numpy as np
from azure.search.documents.models import VectorizedQuery
from loguru import logger

from mmct.providers.base import BaseKeyframesVectorDBProvider, BaseImageEmbeddingProvider


class KeyframeSearcher:
    """Search for keyframes using text queries with injected providers."""

    def __init__(self,
                 search_provider: BaseKeyframesVectorDBProvider,
                 image_embedding_provider: BaseImageEmbeddingProvider,
                 provider_config: Optional[dict] = None):
        """
        Initialize the keyframe searcher with injected providers.

        Args:
            search_provider: BaseKeyframesVectorDBProvider instance for vector search
            image_embedding_provider: BaseImageEmbeddingProvider for generating query embeddings
            provider_config: Optional provider configuration overrides
        """
        self.search_provider = search_provider
        self.image_embedding_provider = image_embedding_provider
        
        # Apply provider config overrides if provided
        if provider_config and hasattr(self.search_provider, 'config'):
            try:
                self.search_provider.config.update(provider_config)
            except Exception:
                self.search_provider.config = provider_config


    async def search_keyframes(self,
                        query: str,
                        top_k: int = 5,
                        video_filter: Optional[Dict] = None
                        ) -> List[Dict[str, Any]]:
        """
        Search for keyframes using text query.

        Args:
            query: Text query to search for
            top_k: Number of results to return
            video_filter: Optional filter for specific video

        Returns:
            List of search results
        """
        try:
            # Generate query embedding using image embedding provider
            query_embedding = await self.image_embedding_provider.text_embedding(query)
            
            # Convert to list if numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
        
            # Create vector query for search
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="embeddings"
            )

            # Perform search using provider
            results = await self.search_provider.search(
                query=query if query else "*",
                search_text=query if query else "*",
                vector_queries=[vector_query],
                embedding=query_embedding,
                filter=video_filter,
                top=top_k,
                query_type="vector",
            )
            fields_to_retrieve=["keyframe_filename", "video_id", "timestamp_seconds", "motion_score"]

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
            raise Exception(f"An error occurred while fetching keyframe from index: {str(e)}") from e
