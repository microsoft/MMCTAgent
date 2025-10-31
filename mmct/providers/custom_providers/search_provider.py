from mmct.providers.base import SearchProvider
from typing import Dict, Any, List
from mmct.utils.error_handler import ProviderException
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from loguru import logger
from mmct.providers.custom_providers.graph_rag.document_graph_handler import DocumentGraph


class CustomSearchProvider(SearchProvider):
    """Custom Search provider implementation using GraphRAG."""

    def __init__(self, config: Dict[str, Any]):
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Custom Search client."""
        try:
            self.client = DocumentGraph()
        except Exception as e:
            raise ProviderException(f"Failed to initialize Custom Search client: {e}")

    # ------------------------- SEARCH ------------------------- #
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """Search documents using GraphRAG."""
        try:
            embedding = kwargs.get("embedding")
            top_k = kwargs.get("top_k", 5)
            top_n = kwargs.get("top_n", 3)

            if embedding is None:
                raise ProviderException("Embedding not provided for user query")

            results = self.client.search(
                query_embedding=embedding,
                top_k=top_k,
                top_n=top_n
            )
            return results
        except Exception as e:
            logger.error(f"Custom Search failed: {e}")
            raise ProviderException(f"Custom Search failed: {e}")

    # ------------------------- INDEX ------------------------- #
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict, index_name: str = None) -> bool:
        try:

            if not document:
                raise ProviderException(
                    "Document must not be empty" 
                )

            logger.info(f"Indexing video_id={document[0]['hash_video_id']} with {len(document)} chapters")
            self.client.add_documents(video_documents=document, video_id=document[0]['hash_video_id'])
            logger.success(f"Successfully indexed video_id={document[0]['hash_video_id']}")
            return True
        except Exception as e:
            logger.error(f"Custom Search indexing failed: {e}")
            raise ProviderException(f"Custom Search indexing failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str, index_name: str = None) -> bool:
        """Delete a document from GraphRAG."""
        logger.warning("Delete document not implemented for CustomSearchProvider (GraphRAG)")
        return False

    async def create_index(self, index_name: str, index_schema: Any) -> bool:
        """
        Index creation not applicable for GraphRAG.
        GraphRAG manages its own internal graph structure.
        """
        logger.info("Index creation not applicable for GraphRAG - using internal graph structure")
        return True

    async def index_exists(self, index_name: str) -> bool:
        """
        Index existence check not applicable for GraphRAG.
        GraphRAG manages its own internal graph structure.
        """
        return True

    async def delete_index(self, index_name: str) -> bool:
        """
        Index deletion not applicable for GraphRAG.
        GraphRAG manages its own internal graph structure.
        """
        logger.warning("Index deletion not applicable for GraphRAG")
        return False