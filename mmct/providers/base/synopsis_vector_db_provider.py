"""Abstract base class for synopsis vector database providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from mmct.providers.search_document_models import SynopsisIndexDocument


class BaseSynopsisVectorDBProvider(ABC):
    """Abstract base class for synopsis vector database providers."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
    
    @abstractmethod
    def get_index_schema(self) -> Any:
        """
        Creates provider-specific schema based on SynopsisIndexDocument type.
        
        Returns:
            Provider-specific index schema object
        """
        pass
    
    @abstractmethod
    def parse_response(self, vector_db_document: Any) -> SynopsisIndexDocument:
        """
        Parses the retrieved vector DB document into SynopsisIndexDocument object.
        
        Args:
            vector_db_document: Provider-specific document response
            
        Returns:
            SynopsisIndexDocument: Parsed document
        """
        pass
    
    @abstractmethod
    async def upload_synopsis(self, synopsis: SynopsisIndexDocument) -> bool:
        """
        Upload a synopsis document to the index.
        
        Args:
            synopsis: Synopsis document to upload
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def search_synopsis(self, video_id: str) -> Optional[SynopsisIndexDocument]:
        """
        Retrieve synopsis for a video by ID.
        
        Args:
            video_id: Video identifier
            
        Returns:
            SynopsisIndexDocument if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def semantic_search_synopsis(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[SynopsisIndexDocument]:
        """
        Semantic search across all synopses.
        
        Args:
            query: Search query text
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of matching synopsis documents
        """
        pass
    
    @abstractmethod
    async def delete_synopsis(self, video_id: str) -> bool:
        """
        Delete synopsis for a video.
        
        Args:
            video_id: Video identifier
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def create_index(self) -> bool:
        """
        Create the search index with the given schema.
        
        Returns:
            bool: True if created, False if already exists
        """
        pass
    
    @abstractmethod
    async def index_exists(self) -> bool:
        """
        Check if an index exists.
        
        Returns:
            bool: True if index exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_index(self) -> bool:
        """
        Delete the search index.
        
        Returns:
            bool: True if successful
        """
        pass
