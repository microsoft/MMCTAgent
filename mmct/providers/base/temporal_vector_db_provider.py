"""Abstract base class for temporal events vector database providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from mmct.providers.search_document_models import TemporalEventIndexDocument


class BaseTemporalVectorDBProvider(ABC):
    """Abstract base class for temporal events vector database providers."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
    
    @abstractmethod
    def get_index_schema(self) -> Any:
        """
        Creates provider-specific schema based on TemporalEventIndexDocument type.
        
        Returns:
            Provider-specific index schema object
        """
        pass
    
    @abstractmethod
    def parse_response(self, vector_db_document: Any) -> TemporalEventIndexDocument:
        """
        Parses the retrieved vector DB document into TemporalEventIndexDocument object.
        
        Args:
            vector_db_document: Provider-specific document response
            
        Returns:
            TemporalEventIndexDocument: Parsed document
        """
        pass
    
    @abstractmethod
    async def upload_events(self, events: List[TemporalEventIndexDocument]) -> bool:
        """
        Batch upload temporal events.
        
        Args:
            events: List of event documents to upload
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def search_by_time_range(
        self,
        video_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[TemporalEventIndexDocument]:
        """
        Search events within a time range.
        
        Args:
            video_id: Video identifier
            start_time: Start timestamp in seconds (optional)
            end_time: End timestamp in seconds (optional)
            event_type: Filter by event type (optional)
            limit: Maximum number of results
            
        Returns:
            List of matching event documents
        """
        pass
    
    @abstractmethod
    async def get_events_in_sequence(
        self,
        video_id: str,
        start_sequence: int,
        end_sequence: int,
    ) -> List[TemporalEventIndexDocument]:
        """
        Get events by sequence number range.
        
        OPTIMIZATION: Single query to fetch all events in range,
        enabling efficient chain traversal without N+1 queries.
        
        Args:
            video_id: Video identifier
            start_sequence: Start sequence number (inclusive)
            end_sequence: End sequence number (inclusive)
            
        Returns:
            List of events in sequence order
        """
        pass
    
    @abstractmethod
    async def get_events_by_relationship(
        self,
        target_event_id: str,
        relationship: str,  # "precedes" or "follows"
        limit: int = 10,
    ) -> List[TemporalEventIndexDocument]:
        """
        Get events by relationship using indexed filters.
        
        Uses filterable array fields for O(1) lookup:
        - precedes_event_ids/any(id: id eq 'target')
        - follows_event_ids/any(id: id eq 'target')
        
        Args:
            target_event_id: Target event identifier
            relationship: Relationship type ("precedes" or "follows")
            limit: Maximum number of results
            
        Returns:
            List of related events
        """
        pass
    
    @abstractmethod
    async def semantic_search_events(
        self,
        query: str,
        query_vector: List[float],
        video_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[TemporalEventIndexDocument]:
        """
        Semantic search across event descriptions.
        
        Args:
            query: Search query text
            query_vector: Query embedding vector
            video_id: Filter by video ID (optional)
            top_k: Number of results to return
            
        Returns:
            List of matching event documents
        """
        pass
    
    @abstractmethod
    async def delete_events(self, video_id: str) -> bool:
        """
        Delete all events for a video.
        
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
