"""Abstract base class for object vector database providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict


class BaseObjectVectorDBProvider(ABC):
    """Abstract base class for object vector database providers.
    
    Provides interface for storing, indexing, and searching object documents
    with vector embeddings for semantic similarity search.
    """
    
    def __init__(self, index_name: str):
        """Initialize the object vector database provider.
        
        Args:
            index_name: Name of the index to use for operations.
        """
        self.index_name = index_name
    
    @abstractmethod
    async def search(
        self,
        query: str,
        video_id: Optional[str] = None,
        object_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search objects by text query.
        
        Args:
            query: Text query for searching objects.
            video_id: Optional video ID to filter results.
            object_type: Optional object type filter.
            limit: Maximum number of results to return.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            List of matching object documents with scores.
        """
        pass
    
    @abstractmethod
    async def search_by_vector(
        self,
        query_vector: List[float],
        video_id: Optional[str] = None,
        object_type: Optional[str] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search objects by vector similarity.
        
        Args:
            query_vector: Query embedding vector for similarity search.
            video_id: Optional video ID to filter results.
            object_type: Optional object type filter.
            limit: Maximum number of results to return.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            List of matching object documents with similarity scores.
        """
        pass
    
    @abstractmethod
    async def search_similar_objects(
        self,
        object_id: str,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Find objects similar to a given object.
        
        Args:
            object_id: ID of the reference object.
            limit: Maximum number of similar objects to return.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            List of similar object documents with similarity scores.
        """
        pass
    
    @abstractmethod
    async def index_document(
        self,
        document: Dict[str, Any],
        **kwargs
    ) -> bool:
        """Index a single object document.
        
        Args:
            document: Object document to index. Must contain 'id' field
                and optionally 'embedding_vector' for vector search.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            True if indexing succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    async def create_index(
        self,
        schema: Optional[Any] = None,
        **kwargs
    ) -> bool:
        """Create the search index.
        
        Args:
            schema: Optional schema definition for the index.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            True if index was created or already exists.
        """
        pass
    
    @abstractmethod
    async def delete_document(
        self,
        document_id: str,
        **kwargs
    ) -> bool:
        """Delete a document from the index.
        
        Args:
            document_id: ID of the document to delete.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            True if deletion succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    async def index_exists(self) -> bool:
        """Check if the index exists.
        
        Returns:
            True if index exists, False otherwise.
        """
        pass
    
    @abstractmethod
    async def delete_index(self) -> bool:
        """Delete the search index.
        
        Returns:
            True if deletion succeeded, False otherwise.
        """
        pass
    
    async def batch_index_documents(
        self,
        documents: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Batch index multiple object documents.
        
        Default implementation iterates over documents.
        Subclasses may override for optimized batch operations.
        
        Args:
            documents: List of object documents to index.
            **kwargs: Additional provider-specific parameters.
            
        Returns:
            Dictionary with 'success' count and 'failed' count.
        """
        success_count = 0
        failed_count = 0
        
        for doc in documents:
            try:
                if await self.index_document(doc, **kwargs):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
        
        return {"success": success_count, "failed": failed_count}
    
    async def close(self) -> None:
        """Close provider connections and cleanup resources.
        
        Subclasses should override to perform cleanup operations.
        """
        pass
