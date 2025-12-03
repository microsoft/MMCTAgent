from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Type
from mmct.providers.search_document_models import ObjectCollectionDocument

class BaseObjectCollectionVectorDBProvider(ABC):
    """Abstract base class for Object Collection - Vector DB providers."""

    def __init__(self, index_name):
        self.index_name = index_name

    @abstractmethod
    def get_index_schema(self) -> Any:
        """
        Creates provider-specific schema based on ObjectCollectionDocument type.

        Returns:
            Provider-specific index schema object
        """
        pass

    @abstractmethod
    def parse_response(self, vector_db_document: Any) -> ObjectCollectionDocument:
        """
        Parses the retrieved vector DB document into ObjectCollectionDocument object.

        Args:
            vector_db_document: Provider-specific document response

        Returns:
            ObjectCollectionDocument: Parsed document
        """
        pass

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[Tuple[ObjectCollectionDocument, float]]:
        """
        Search for documents.

        Args:
            query: Search query string
            **kwargs: Additional provider-specific search parameters

        Returns:
            List of tuples containing (ObjectCollectionDocument, similarity_score)
        """
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
    async def create_index(self) -> bool:
        """
        Create a index with the given schema.

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
        Delete a index.

        Returns:
            bool: True if successful
        """
        pass

    @abstractmethod
    async def upload_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """
        Upload multiple documents to the index.

        Args:
            documents: List of document dictionaries to upload

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

        Returns:
            bool: True if document exists, False otherwise
        """
        pass

    async def close(self):
        """Close the client and cleanup resources. Optional to implement."""
        pass
