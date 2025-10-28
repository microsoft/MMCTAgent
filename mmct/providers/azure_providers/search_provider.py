
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException
from loguru import logger
from typing import Dict, Any, List
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from mmct.providers.base import SearchProvider
from mmct.providers.credentials import AzureCredentials

class AzureSearchProvider(SearchProvider):
    """Azure AI Search provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_async_credentials()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure AI Search client."""
        try:
            endpoint = self.config.get("endpoint")
            index_name = self.config.get("index_name", "default")
            use_managed_identity = self.config.get("use_managed_identity", True)
            
            if not endpoint:
                raise ConfigurationException("Azure AI Search endpoint is required")
            
            if use_managed_identity:
                return SearchClient(
                    endpoint=endpoint,
                    index_name=index_name,
                    credential=self.credential
                )
            else:
                api_key = self.config.get("api_key")
                if not api_key:
                    raise ConfigurationException("Azure AI Search API key is required when managed identity is disabled")
                
                from azure.core.credentials import AzureKeyCredential
                return SearchClient(
                    endpoint=endpoint,
                    index_name=index_name,
                    credential=AzureKeyCredential(api_key)
                )
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure AI Search client: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """Search documents using Azure AI Search."""
        try:
            vector_queries = None
            semantic_configuration_name=None

            search_text = kwargs.pop("search_text", query)
            top = kwargs.pop("top", None)
            embedding = kwargs.pop("embedding", [])
            query_type = kwargs.pop("query_type", None)
            vector_queries = kwargs.pop("vector_queries",None)

            if query_type=="semantic":
                semantic_configuration_name=kwargs.pop("semantic_configuration_name","my-semantic-search-config")
                search_text = None
                
            if query_type=="vector":
                query_type = None
                
            if embedding and top and not vector_queries:
                vector_query = VectorizedQuery(
                    vector=embedding, k_nearest_neighbors=top, fields="embeddings"
                )
                vector_queries = [vector_query]

            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client
            
            results = await client.search(
                search_text=search_text,
                top=top,
                query_type=query_type,
                vector_queries=vector_queries,
                semantic_configuration_name=semantic_configuration_name,
                **kwargs
            )
            
            return [dict(result) async for result in results]
        except Exception as e:
            logger.error(f"Azure AI Search failed: {e}")
            raise ProviderException(f"Azure AI Search failed: {e}")
        
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict, index_name: str = None) -> bool:
        """Index a document in Azure AI Search."""
        try:
            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client
            
            result = await client.upload_documents(documents=[document])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search indexing failed: {e}")
            raise ProviderException(f"Azure AI Search indexing failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str, index_name: str = None) -> bool:
        """Delete a document from Azure AI Search."""
        try:
            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client
            
            result = await client.delete_documents(documents=[{"id": doc_id}])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search deletion failed: {e}")
            raise ProviderException(f"Azure AI Search deletion failed: {e}")

