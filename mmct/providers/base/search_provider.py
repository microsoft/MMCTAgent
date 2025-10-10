from abc import ABC, abstractmethod
from typing import Dict, List

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
