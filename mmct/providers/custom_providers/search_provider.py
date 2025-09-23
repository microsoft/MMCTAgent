from mmct.providers.base import SearchProvider
from typing import Dict, Any, List
from mmct.exceptions import ProviderException
from mmct.utils.error_handler import handle_exceptions, convert_exceptions 
from loguru import logger
from mmct.providers.custom_providers.graph_rag.document_graph_handler import DocumentGraph

class CustomSearchProvider(SearchProvider):
    """Custom Search provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Custom Search client."""
        try:
            self.client = DocumentGraph()       
        except Exception as e:
            raise ProviderException(f"Failed to initialize Custom Search client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """Search documents using Custom Search."""
        try:
            embedding = kwargs.get('embedding')
            top_k = kwargs.get('top_k')
            top_n = kwargs.get('top_n')
            if not embedding:
                raise ProviderException("embedding not provided for user query")

            if not top_k:
                logger.warning("Top K not provided to Custom Search. Setting it to 5.")
                top_k = 5
            
            if not top_n:
                logger.warning("TOP N (neighbors) not provided to Custom Search. Setting it to 3.")
                top_n = 3

            results = self.client.search(
                query_embedding=embedding,
                top_k=top_k,
                top_n=top_n
            )

            return results
            
        except Exception as e:
            logger.error(f"Custom Search failed: {e}")
            raise ProviderException(f"Custom Search failed: {e}")
        
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict, index_name: str = None) -> bool:
        """Index a document in Custom Search."""
        try:
            pass
        except Exception as e:
            logger.error(f"Custom Search indexing failed: {e}")
            raise ProviderException(f"Custom Search indexing failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str, index_name: str = None) -> bool:
        """Delete a document from Custom Search."""
        try:
            pass
        except Exception as e:
            logger.error(f"Custom Search deletion failed: {e}")
            raise ProviderException(f"Custom Search deletion failed: {e}")

