"""
Azure AI Search client for visual keyframe search using CLIP embeddings.

This module provides:
1. CLIP text embedding generation for queries
2. Vector similarity search in Azure AI Search
3. Retrieval of keyframe metadata (timestamps, URLs, etc.)
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


class VisualSearchClient:
    """Client for searching video keyframes in Azure AI Search using CLIP embeddings."""

    def __init__(
        self,
        search_endpoint: str,
        index_name: str,
        search_api_key: Optional[str] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        """
        Initialize the visual search tool.

        Args:
            search_endpoint: Azure AI Search endpoint URL
            index_name: Name of the search index
            search_api_key: Optional API key for Azure Search
            clip_model_name: CLIP model to use for embeddings
        """
        self.search_endpoint = search_endpoint
        self.index_name = index_name
        self.clip_model_name = clip_model_name

        # Initialize CLIP model
        self._initialize_clip_model()

        # Initialize search client
        self._initialize_search_client(search_api_key)

    def _initialize_clip_model(self):
        """Initialize the CLIP model and processor."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing CLIP model {self.clip_model_name} on {device}")

            self.model = CLIPModel.from_pretrained(self.clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.model = self.model.to(device)
            self.model.eval()
            self.device = device

            logger.info("CLIP model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CLIP model: {e}")
            raise

    def _initialize_search_client(self, api_key: Optional[str] = None):
        """Initialize Azure AI Search client."""
        try:
            from ingestion.auth import get_azure_credential

            # Use API key if provided, otherwise use Azure credentials
            if api_key:
                credential = AzureKeyCredential(api_key)
            else:
                credential = get_azure_credential()

            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=credential
            )
            logger.info("Search client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize search client: {e}")
            raise

    def embed_text_query(self, query: str) -> np.ndarray:
        """
        Generate CLIP embedding for a text query.

        Args:
            query: Text query to embed

        Returns:
            Normalized embedding vector
        """
        try:
            # Process text
            inputs = self.processor(text=[query], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize embedding
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Convert to numpy
            embedding = text_features.cpu().numpy()[0]

            logger.info(f"Generated embedding for query: '{query}'")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise

    def search_keyframes(
        self,
        query: str,
        top_k: int = 10,
        video_id: Optional[str] = None,
        youtube_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for keyframes matching the text query.

        Args:
            query: Text query to search for
            top_k: Number of top results to return
            video_id: Optional video ID to filter results
            youtube_url: Optional YouTube URL to filter results. Takes precedence over video_id.

        Returns:
            List of matching keyframes with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_text_query(query)

            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding.tolist(),
                k_nearest_neighbors=top_k,
                fields="clip_embedding"
            )

            # Build search parameters
            search_params = {
                "vector_queries": [vector_query],
                "top": top_k,
                "select": [
                    "id",
                    "video_id",
                    "keyframe_filename",
                    "timestamp_seconds",
                    "motion_score",
                    "blob_url",
                    "youtube_url",
                    "created_at"
                ]
            }

            # Add filter: youtube_url takes precedence over video_id
            if youtube_url:
                search_params["filter"] = f"youtube_url eq '{youtube_url}'"
            elif video_id:
                search_params["filter"] = f"video_id eq '{video_id}'"

            # Execute search
            results = self.search_client.search(**search_params)

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "video_id": result.get("video_id"),
                    "keyframe_filename": result.get("keyframe_filename"),
                    "timestamp_seconds": result.get("timestamp_seconds"),
                    "motion_score": result.get("motion_score"),
                    "blob_url": result.get("blob_url"),
                    "youtube_url": result.get("youtube_url"),
                    "created_at": result.get("created_at"),
                    "search_score": result.get("@search.score", 0.0)
                })

            logger.info(f"Found {len(formatted_results)} keyframes for query: '{query}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search keyframes: {e}")
            raise

# Standalone function for easy import
def search_keyframes(
    query: str,
    search_endpoint: str,
    index_name: str,
    search_api_key: Optional[str] = None,
    top_k: int = 10,
    video_id: Optional[str] = None,
    youtube_url: Optional[str] = None,
    clip_model_name: str = "openai/clip-vit-base-patch32"
) -> Dict[str, Any]:
    """
    Search for keyframes matching a text query using Azure AI Search.

    Args:
        query: Text query describing what to search for
        search_endpoint: Azure AI Search endpoint URL
        index_name: Name of the search index
        search_api_key: Optional API key for Azure Search
        top_k: Number of top results to return (default: 10)
        video_id: Optional video ID to filter results
        youtube_url: Optional YouTube URL to filter results. Takes precedence over video_id.
        clip_model_name: CLIP model to use (default: openai/clip-vit-base-patch32)

    Returns:
        Dictionary with query results including timestamps and blob URLs
    """
    client = VisualSearchClient(
        search_endpoint=search_endpoint,
        index_name=index_name,
        search_api_key=search_api_key,
        clip_model_name=clip_model_name
    )

    keyframes = client.search_keyframes(query, top_k, video_id, youtube_url)

    return {
        "query": query,
        "total_results": len(keyframes),
        "keyframes": keyframes
    }
