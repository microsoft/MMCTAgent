"""Base provider for chapter group vector database operations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseGroupVectorDBProvider(ABC):
    """Abstract base class for chapter group vector database providers.
    
    Provides interface for storing and querying chapter groups with
    support for order-based filtering and sorting.
    """
    
    @abstractmethod
    async def search_by_vector(
        self,
        query_vector: List[float],
        video_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search groups by vector similarity.
        
        Args:
            query_vector: Query embedding vector.
            video_id: Optional video ID filter.
            limit: Maximum results to return.
            min_score: Minimum similarity score threshold.
            
        Returns:
            List of matching group documents.
        """
        pass
    
    @abstractmethod
    async def get_groups_by_video(
        self,
        video_id: str,
        order_by: str = "order",
        ascending: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get all groups for a video, ordered by specified field.
        
        Args:
            video_id: Video ID to filter by.
            order_by: Field to order by (default: "order").
            ascending: Sort direction (default: True for ascending).
            
        Returns:
            List of group documents in order.
        """
        pass
    
    @abstractmethod
    async def get_group_by_order(
        self,
        video_id: str,
        order: int,
    ) -> Optional[Dict[str, Any]]:
        """Get a specific group by video and order index.
        
        Args:
            video_id: Video ID.
            order: Group order index (0-indexed).
            
        Returns:
            Group document or None if not found.
        """
        pass
    
    @abstractmethod
    async def get_adjacent_groups(
        self,
        video_id: str,
        current_order: int,
        direction: str = "both",
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get adjacent groups (previous and/or next).
        
        Args:
            video_id: Video ID.
            current_order: Current group's order index.
            direction: "previous", "next", or "both".
            
        Returns:
            Dictionary with "previous" and/or "next" group documents.
        """
        pass
    
    @abstractmethod
    async def index_document(
        self,
        document: Dict[str, Any],
    ) -> bool:
        """Index a group document with order metadata.
        
        Args:
            document: Group document with id, video_id, order, summary, embedding.
            
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    async def get_total_groups(
        self,
        video_id: str,
    ) -> int:
        """Get total number of groups for a video.
        
        Args:
            video_id: Video ID.
            
        Returns:
            Total count of groups.
        """
        pass
    
    @abstractmethod
    async def create_index(
        self,
        dimension: int,
        index_name: str = "groups",
    ) -> bool:
        """Create the group index.
        
        Args:
            dimension: Embedding dimension.
            index_name: Name of the index.
            
        Returns:
            True if successful.
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close provider connections."""
        pass
