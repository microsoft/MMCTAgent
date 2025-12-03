from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException
from loguru import logger
from typing import Dict, Any, List, Optional, Union, Tuple
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import SearchIndex
from mmct.providers.base.keyframes_vector_db_provider import BaseKeyframesVectorDBProvider
from mmct.providers.search_document_models import KeyframeDocument
from mmct.providers.azure_providers.azure_schema_utils import (
    create_azure_index_schema,
    parse_azure_response_to_model,
    extract_similarity_score,
)


class AISearchKeyframesProvider(BaseKeyframesVectorDBProvider):
    """Azure AI Search provider implementation for keyframes."""

    def __init__(
        self,
        index_name: str,
        endpoint: str,
        credentials: Optional[Union[AzureKeyCredential, AsyncTokenCredential]] = None,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        if not endpoint:
            raise ConfigurationException("Azure AI Search endpoint is required!")

        if not index_name:
            raise ConfigurationException("index name is required for indexing!")

        # Validate that exactly one of credentials or api_key is provided
        if credentials is None and api_key is None:
            raise ConfigurationException("Either credentials or api_key must be provided!")

        if credentials is not None and api_key is not None:
            raise ConfigurationException("Only one of credentials or api_key should be provided, not both!")

        self.credentials = credentials
        self.api_key = api_key
        self.index_name = index_name
        self.endpoint = endpoint
        self.dimensions = dimensions
        self.index_client = self._initialize_index_client()

        # Cache for search client
        self._search_client: Optional[SearchClient] = None

    def _get_search_client(self) -> SearchClient:
        """Get or create the SearchClient for this provider's index."""
        if self._search_client is not None:
            return self._search_client

        try:
            if self.credentials is not None:
                self._search_client = SearchClient(
                    endpoint=self.endpoint, index_name=self.index_name, credential=self.credentials
                )
            else:
                self._search_client = SearchClient(
                    endpoint=self.endpoint,
                    index_name=self.index_name,
                    credential=AzureKeyCredential(self.api_key),
                )
            return self._search_client
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure AI Search client: {e}")

    def _initialize_index_client(self) -> SearchIndexClient:
        """Initialize Azure AI Search Index client for index management."""
        try:
            if self.credentials is not None:
                return SearchIndexClient(endpoint=self.endpoint, credential=self.credentials)
            else:
                return SearchIndexClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.api_key))
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure AI Search Index client: {e}")

    def get_index_schema(self) -> SearchIndex:
        """Creates Azure AI Search specific schema based on KeyframeDocument type."""
        return create_azure_index_schema(
            model_class=KeyframeDocument,
            index_name=self.index_name,
            vector_dimensions=self.dimensions or 1536,
            vector_field_name="embeddings",
        )

    def parse_response(self, vector_db_document: Any) -> KeyframeDocument:
        """Parses the retrieved Azure vector DB document into KeyframeDocument object."""
        return parse_azure_response_to_model(vector_db_document, KeyframeDocument)

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, **kwargs) -> List[Tuple[KeyframeDocument, float]]:
        """Search documents using Azure AI Search."""
        try:
            search_text = kwargs.pop("search_text", query)
            top = kwargs.pop("top", None)
            embedding = kwargs.pop("embedding", [])
            query_type = kwargs.pop("query_type", None)
            vector_queries = kwargs.pop("vector_queries", None)
            embedding_field_name = kwargs.pop("embedding_field_name", "embeddings")
            filters = kwargs.pop("filter", None)
            semantic_configuration_name = None

            if filters:
                kwargs["filter"] = await self._build_filter_query(filters=filters)

            if query_type == "semantic":
                semantic_configuration_name = kwargs.pop(
                    "semantic_configuration_name", "my-semantic-search-config"
                )
                search_text = None

            if query_type == "vector":
                query_type = None
                search_text = None

            if embedding and top and not vector_queries:
                vector_query = VectorizedQuery(
                    vector=embedding, k_nearest_neighbors=top, fields=embedding_field_name
                )
                vector_queries = [vector_query]

            client = self._get_search_client()
            results = await client.search(
                search_text=search_text,
                top=top,
                query_type=query_type,
                vector_queries=vector_queries,
                semantic_configuration_name=semantic_configuration_name,
                **kwargs,
            )

            parsed_results = []
            async for result in results:
                result_dict = dict(result)
                document = self.parse_response(result_dict)
                score = extract_similarity_score(result_dict)
                parsed_results.append((document, score))

            return parsed_results
        except Exception as e:
            logger.error(f"Azure AI Search failed: {e}")
            raise ProviderException(f"Azure AI Search failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_document(self, document: Dict) -> bool:
        """Index a document in Azure AI Search."""
        try:
            client = self._get_search_client()
            result = await client.upload_documents(documents=[document])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search indexing failed: {e}")
            raise ProviderException(f"Azure AI Search indexing failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from Azure AI Search."""
        try:
            client = self._get_search_client()
            result = await client.delete_documents(documents=[{"id": doc_id}])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search deletion failed: {e}")
            raise ProviderException(f"Azure AI Search deletion failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def create_index(self) -> bool:
        """Create a search index using the schema from get_index_schema()."""
        try:
            index_schema = self.get_index_schema()
            await self.index_client.create_index(index_schema)
            logger.info(f"Successfully created index '{self.index_name}'")
            return True
        except Exception as e:
            if "ResourceNameAlreadyInUse" in str(e) or "already exists" in str(e):
                logger.info(f"Index '{self.index_name}' already exists")
                return False
            else:
                logger.error(f"Failed to create index '{self.index_name}': {e}")
                raise ProviderException(f"Failed to create index '{self.index_name}': {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_exists(self) -> bool:
        """Check if an index exists."""
        try:
            await self.index_client.get_index(self.index_name)
            return True
        except Exception as e:
            error_str = str(e)
            not_found_patterns = [
                "ResourceNotFound", "NotFound", "does not exist",
                "was not found", "No index with the name",
            ]
            if any(pattern in error_str for pattern in not_found_patterns):
                return False
            else:
                logger.error(f"Error checking if index '{self.index_name}' exists: {e}")
                raise ProviderException(f"Error checking if index '{self.index_name}' exists: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_index(self) -> bool:
        """Delete a search index."""
        try:
            await self.index_client.delete_index(self.index_name)
            logger.info(f"Successfully deleted index '{self.index_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index '{self.index_name}': {e}")
            raise ProviderException(f"Failed to delete index '{self.index_name}': {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """Upload multiple documents to the search index."""
        try:
            client = self._get_search_client()
            result = await client.upload_documents(documents=documents)
            logger.info(f"Successfully uploaded {len(documents)} documents to index")
            return {"success": True, "count": len(documents), "result": result}
        except Exception as e:
            logger.error(f"Azure AI Search bulk upload failed: {e}")
            raise ProviderException(f"Azure AI Search bulk upload failed: {e}")

    async def _build_filter_query(self, filters: Dict[str, Any]) -> str:
        """Build filter query string from a dictionary of filters."""
        expressions = []
        for field, ops in filters.items():
            for op, value in ops.items():
                if isinstance(value, str):
                    value_str = f"'{value}'"
                else:
                    value_str = str(value)
                expressions.append(f"{field} {op} {value_str}")
        return " and ".join(expressions)

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def check_is_document_exist(self, hash_id: str) -> bool:
        """Check if a document with the given hash_id exists in the index."""
        try:
            client = self._get_search_client()
            results = await client.search(
                search_text="*", filter=f"video_id eq '{hash_id}'", top=1
            )
            async for _ in results:
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to check if document exists: {e}")
            raise ProviderException(f"Failed to check if document exists: {e}")

    async def close(self):
        """Close the search client and cleanup resources."""
        try:
            if self._search_client:
                logger.info(f"Closing Azure AI Search client for index '{self.index_name}'")
                await self._search_client.close()
            if self.index_client:
                logger.info("Closing Azure AI Search Index client")
                await self.index_client.close()
        except Exception as e:
            logger.error(f"Error during client cleanup: {e}")
