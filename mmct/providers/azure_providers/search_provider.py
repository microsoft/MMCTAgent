
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException
from loguru import logger
from typing import Dict, Any, List
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import SearchIndex
from mmct.providers.base import SearchProvider
from mmct.providers.credentials import AzureCredentials

class AzureSearchProvider(SearchProvider):
    """Azure AI Search provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_async_credentials()
        self.client = self._initialize_client()
        self.index_client = self._initialize_index_client()
    
    def _initialize_client(self):
        """Initialize Azure AI Search client."""
        try:
            endpoint = self.config.get("endpoint")
            if not endpoint:
                raise ConfigurationException("SEARCH_ENDPOINT environment variable not set")

            index_name = self.config.get("index_name", "default")
            use_managed_identity = self.config.get("use_managed_identity", True)
            
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

    def _initialize_index_client(self):
        """Initialize Azure AI Search Index client for index management."""
        try:
            endpoint = self.config.get("endpoint")
            if not endpoint:
                raise ConfigurationException("Azure AI Search endpoint is required")

            return SearchIndexClient(endpoint=endpoint, credential=self.credential)
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure AI Search Index client: {e}")

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

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def create_index(self, index_name: str, index_schema: SearchIndex) -> bool:
        """
        Create a search index with the given schema.

        Args:
            index_name: Name of the index to create
            index_schema: Azure SearchIndex object defining the index schema

        Returns:
            bool: True if created, False if already exists
        """
        try:
            await self.index_client.create_index(index_schema)
            logger.info(f"Successfully created index '{index_name}'")
            return True
        except Exception as e:
            if "ResourceNameAlreadyInUse" in str(e) or "already exists" in str(e):
                logger.info(f"Index '{index_name}' already exists")
                return False
            else:
                logger.error(f"Failed to create index '{index_name}': {e}")
                raise ProviderException(f"Failed to create index '{index_name}': {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if index exists, False otherwise
        """
        try:
            await self.index_client.get_index(index_name)
            return True
        except Exception as e:
            error_str = str(e)
            # Check for various "index not found" error patterns
            not_found_patterns = [
                "ResourceNotFound",
                "NotFound",
                "does not exist",
                "was not found",
                "No index with the name"
            ]
            if any(pattern in error_str for pattern in not_found_patterns):
                return False
            else:
                logger.error(f"Error checking if index '{index_name}' exists: {e}")
                raise ProviderException(f"Error checking if index '{index_name}' exists: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete a search index.

        Args:
            index_name: Name of the index to delete

        Returns:
            bool: True if successful
        """
        try:
            await self.index_client.delete_index(index_name)
            logger.info(f"Successfully deleted index '{index_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index '{index_name}': {e}")
            raise ProviderException(f"Failed to delete index '{index_name}': {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_documents(self, documents: List[Dict], index_name: str = None) -> Dict[str, Any]:
        """
        Upload multiple documents to the search index.

        Args:
            documents: List of document dictionaries to upload
            index_name: Optional index name (uses default if not provided)

        Returns:
            Dict with upload results
        """
        try:
            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client

            result = await client.upload_documents(documents=documents)
            logger.info(f"Successfully uploaded {len(documents)} documents to index")
            return {"success": True, "count": len(documents), "result": result}
        except Exception as e:
            logger.error(f"Azure AI Search bulk upload failed: {e}")
            raise ProviderException(f"Azure AI Search bulk upload failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def check_is_document_exist(self, hash_id: str, index_name: str = None) -> bool:
        """
        Check if a document with the given hash_id exists in the index.

        Args:
            hash_id: Hash ID of the document to check
            index_name: Optional index name (uses default if not provided)

        Returns:
            bool: True if document exists, False otherwise
        """
        try:
            if index_name and index_name != self.client._index_name:
                # Create new client for different index
                config = self.config.copy()
                config["index_name"] = index_name
                client = AzureSearchProvider(config).client
            else:
                client = self.client

            # Search for document with the given hash_id
            results = await client.search(
                search_text="*",
                filter=f"hash_video_id eq '{hash_id}'",
                top=1
            )

            # Check if any results were returned
            async for _ in results:
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check if document exists: {e}")
            raise ProviderException(f"Failed to check if document exists: {e}")

    async def close(self):
        """Close the search client and cleanup resources."""
        if self.client:
            logger.info("Closing Azure AI Search client")
            await self.client.close()
        if self.index_client:
            logger.info("Closing Azure AI Search Index client")
            await self.index_client.close()