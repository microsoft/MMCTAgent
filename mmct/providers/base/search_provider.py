from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    async def search(self, query: str, index_name: str, **kwargs) -> List[Dict]:
        """Search for documents."""
        pass

    @abstractmethod
    async def index_document(self, document: Dict, index_name: str) -> bool:
        """Index a document."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str, index_name: str) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    async def create_index(self, index_name: str, index_schema: Any) -> bool:
        """
        Create a search index with the given schema.

        Args:
            index_name: Name of the index to create
            index_schema: Provider-specific index schema definition

        Returns:
            bool: True if created, False if already exists
        """
        pass

    @abstractmethod
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if index exists, False otherwise
        """
        pass

    @abstractmethod
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a search index.

        Args:
            index_name: Name of the index to delete

        Returns:
            bool: True if successful
        """
        pass
