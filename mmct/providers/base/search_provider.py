from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""

    def __init__(self, index_name):
        self.index_name = index_name

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Dict]:
        """Search for documents."""
        pass

    @abstractmethod
    async def index_document(self, document: Dict) -> bool:
        """Index a document."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        pass

    @abstractmethod
    async def create_index(self, index_schema: Any) -> bool:
        """
        Create a search index with the given schema.

        Args:
            index_schema can take one of the predefined schema types like "chapter", "object_registry", "keyframes".

        Returns:
            bool: True if created, False if already exists
        """
        pass

    @abstractmethod
    async def index_exists(self) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if index exists, False otherwise
        """
        pass

    @abstractmethod
    async def delete_index(self) -> bool:
        """
        Delete a search index.

        Args:
            index_name: Name of the index to delete

        Returns:
            bool: True if successful
        """
        pass

    @abstractmethod
    async def upload_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """
        Upload multiple documents to the search index.

        Args:
            documents: List of document dictionaries to upload
            index_name: Optional index name (uses default if not provided)

        Returns:
            Dict with upload results
        """
        pass

    @abstractmethod
    async def check_is_document_exist(self, hash_id: str) -> bool:
        """
        Check if a document with the given hash_id exists in the index.

        Args:
            hash_id: Hash ID of the document to check
            index_name: Optional index name (uses default if not provided)

        Returns:
            bool: True if document exists, False otherwise
        """
        pass

    async def close(self):
        """Close the search client and cleanup resources. Optional to implement."""
        pass
