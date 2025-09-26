import os
import asyncio
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.aio import SearchIndexClient as AsyncSearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import AzureCliCredential, DefaultAzureCredential
import numpy as np


class VideoFrameSearchClient:
    """Azure AI Search client for video frame embeddings and metadata."""
    
    def __init__(self, search_endpoint: str, search_key: Optional[str] = None, 
                 index_name: str = "video-frames-index"):
        """
        Initialize the search client.
        
        Args:
            search_endpoint: Azure AI Search endpoint URL
            search_key: API key (if None, uses Azure CLI credential)
            index_name: Name of the search index
        """
        self.endpoint = search_endpoint
        self.index_name = index_name
        
        # Use API key if provided, otherwise use Azure CLI credential
        if search_key:
            credential = AzureKeyCredential(search_key)
        else:
            try:
                credential = AzureCliCredential()
                credential.get_token("https://search.azure.com/.default")
            except Exception as e:
                credential = DefaultAzureCredential()
        
        self.index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
        self.async_index_client = AsyncSearchIndexClient(endpoint=search_endpoint, credential=credential)
        self.search_client = SearchClient(endpoint=search_endpoint, 
                                        index_name=index_name, 
                                        credential=credential)
        self.async_search_client = AsyncSearchClient(endpoint=search_endpoint, 
                                                   index_name=index_name, 
                                                   credential=credential)
    
    def create_index_if_not_exists(self) -> bool:
        """Create the video frames index if it doesn't exist."""
        try:
            # Check if index exists
            try:
                self.index_client.get_index(self.index_name)
                return True
            except Exception:
                pass
        
            # Define the simplified search index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="video_id", type=SearchFieldDataType.String, 
                              filterable=True, facetable=True),
                SearchableField(name="keyframe_filename", type=SearchFieldDataType.String, 
                              filterable=True, facetable=True),
                SearchField(name="clip_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                           searchable=True, retrievable=True, vector_search_dimensions=512, 
                           vector_search_profile_name="clip-profile"),
                SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, 
                           filterable=True, sortable=True),
                SimpleField(name="motion_score", type=SearchFieldDataType.Double, 
                           filterable=True, sortable=True),
                SimpleField(name="timestamp_seconds", type=SearchFieldDataType.Double, 
                           filterable=True, sortable=True)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="hnsw-algorithm",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="clip-profile",
                        algorithm_configuration_name="hnsw-algorithm"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_search = SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="video-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="video_id"),
                            content_fields=[
                                SemanticField(field_name="keyframe_filename")
                            ],
                            keywords_fields=[
                                SemanticField(field_name="keyframe_filename")
                            ]
                        )
                    )
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            self.index_client.create_index(index)
            return True
            
        except Exception as e:
            return False
    
    def upload_frame_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upload frame documents to the search index.
        
        Args:
            documents: List of frame documents to upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not documents:
                return True
            
            # Upload documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = self.search_client.upload_documents(documents=batch)
                
                # Check for any failed uploads
                failed_count = sum(1 for r in result if not r.succeeded)
                if failed_count > 0:
                    pass
                else:
                    pass
            
            return True
            
        except Exception as e:
            return False
    
    async def search_similar_frames(self, query_vector: np.ndarray, query_text: str = "",
                            top_k: int = 10, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar frames using vector and text search.

        Args:
            query_vector: CLIP embedding vector for similarity search
            query_text: Optional text query for hybrid search
            top_k: Number of results to return
            filters: Optional OData filter expression

        Returns:
            List of search results
        """
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector.tolist(),
                k_nearest_neighbors=top_k,
                fields="clip_embedding"
            )

            # Perform search
            async with self.async_search_client:
                results = await self.async_search_client.search(
                    search_text=query_text if query_text else "*",
                    vector_queries=[vector_query],
                    filter=filters,
                    top=top_k,
                    include_total_count=True
                )

                # Convert results to list
                search_results = []
                async for result in results:
                    search_results.append(dict(result))

                return search_results

        except Exception as e:
            return []
    
    def search_frames_by_video(self, video_id: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Get all frames for a specific video."""
        try:
            filter_expr = f"video_id eq '{video_id}'"
            
            results = self.search_client.search(
                search_text="*",
                filter=filter_expr,
                top=top_k,
                order_by=["created_at asc"]
            )
            
            return [dict(result) for result in results]
            
        except Exception as e:
            return []
    
    def delete_frames_by_video(self, video_id: str) -> bool:
        """Delete all frames for a specific video."""
        try:
            # First, get all frame IDs for this video
            frames = self.search_frames_by_video(video_id, top_k=1000)
            
            if not frames:
                return True
            
            # Delete documents by ID
            documents_to_delete = [{"id": frame["id"]} for frame in frames]
            result = self.search_client.delete_documents(documents=documents_to_delete)
            
            # Check results
            failed_count = sum(1 for r in result if not r.succeeded)
            if failed_count > 0:
                return False
            
            return True
            
        except Exception as e:
            return False


def create_frame_document(video_path: str, frame_path: str, frame_number: int,
                         timestamp_seconds: float, motion_score: float,
                         width: int, height: int, clip_embedding: np.ndarray,
                         extracted_text: str = "", tags: List[str] = None, 
                         video_id: str = None) -> Dict[str, Any]:
    """
    Create a frame document for Azure AI Search with simplified schema.
    
    Args:
        video_path: Path to the source video file
        frame_path: Path to the extracted frame image
        frame_number: Frame number in the video
        timestamp_seconds: Time position in seconds
        motion_score: Optical flow motion score
        width: Frame width in pixels
        height: Frame height in pixels
        clip_embedding: CLIP embedding vector
        extracted_text: Any OCR text found in frame
        tags: Optional tags for the frame
        video_id: Hash-based video ID (if None, uses filename)
        
    Returns:
        Document dictionary ready for Azure AI Search
    """
    import uuid
    from datetime import datetime, timezone
    
    # Generate unique ID
    frame_id = f"{os.path.basename(video_path)}_{frame_number}_{int(timestamp_seconds)}"
    frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, frame_id))
    
    # Use provided video_id (hash) or fallback to filename
    if video_id is None:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
    
    # Generate keyframe filename using video_id
    keyframe_filename = f"{video_id}_{frame_number}.jpg"
    
    document = {
        "id": frame_id,
        "video_id": video_id,
        "keyframe_filename": keyframe_filename,
        "clip_embedding": clip_embedding.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_score": float(motion_score),
        "timestamp_seconds": float(timestamp_seconds)
    }
    
    return document