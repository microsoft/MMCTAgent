"""Azure AI Search provider implementation for synopsis index."""

from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException
from loguru import logger
from typing import Any, List, Optional, Union
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import SearchIndex
from mmct.providers.base.synopsis_vector_db_provider import BaseSynopsisVectorDBProvider
from mmct.providers.search_document_models import SynopsisIndexDocument
from mmct.providers.azure_providers.azure_schema_utils import (
    create_azure_index_schema,
    parse_azure_response_to_model,
)


class AISearchSynopsisProvider(BaseSynopsisVectorDBProvider):
    """Azure AI Search provider implementation for synopsis."""

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
        """Creates Azure AI Search specific schema based on SynopsisIndexDocument type."""
        return create_azure_index_schema(
            model_class=SynopsisIndexDocument,
            index_name=self.index_name,
            vector_dimensions=self.dimensions or 1536,
            vector_field_name="embeddings",
        )

    def parse_response(self, vector_db_document: Any) -> SynopsisIndexDocument:
        """Parses the retrieved Azure vector DB document into SynopsisIndexDocument object."""
        return parse_azure_response_to_model(vector_db_document, SynopsisIndexDocument)

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_synopsis(self, synopsis: SynopsisIndexDocument) -> bool:
        """Upload a synopsis document to the index."""
        try:
            client = self._get_search_client()
            document = synopsis.model_dump()
            
            result = await client.upload_documents(documents=[document])
            
            if result and len(result) > 0:
                if result[0].succeeded:
                    logger.info(f"Successfully uploaded synopsis for video: {synopsis.video_id}")
                    return True
                else:
                    logger.error(f"Failed to upload synopsis: {result[0].error_message}")
                    return False
            return False
        except Exception as e:
            logger.error(f"Failed to upload synopsis: {e}")
            raise ProviderException(f"Failed to upload synopsis: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search_synopsis(self, video_id: str) -> Optional[SynopsisIndexDocument]:
        """Retrieve synopsis for a video by ID."""
        try:
            client = self._get_search_client()
            
            # Search by video_id filter
            results = await client.search(
                search_text="*",
                filter=f"video_id eq '{video_id}'",
                top=1,
            )
            
            async for result in results:
                return self.parse_response(dict(result))
            
            return None
        except Exception as e:
            logger.error(f"Failed to search synopsis: {e}")
            raise ProviderException(f"Failed to search synopsis: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def semantic_search_synopsis(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[SynopsisIndexDocument]:
        """Semantic search across all synopses."""
        try:
            client = self._get_search_client()
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="embeddings"
            )
            
            results = await client.search(
                search_text=query,
                vector_queries=[vector_query],
                top=top_k,
            )
            
            parsed_results = []
            async for result in results:
                document = self.parse_response(dict(result))
                parsed_results.append(document)
            
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to semantic search synopsis: {e}")
            raise ProviderException(f"Failed to semantic search synopsis: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_synopsis(self, video_id: str) -> bool:
        """Delete synopsis for a video."""
        try:
            client = self._get_search_client()
            
            # Delete document by ID (video_id is the key)
            result = await client.delete_documents(documents=[{"video_id": video_id}])
            
            if result and len(result) > 0:
                if result[0].succeeded:
                    logger.info(f"Successfully deleted synopsis for video: {video_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete synopsis: {e}")
            raise ProviderException(f"Failed to delete synopsis: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def create_index(self) -> bool:
        """Create the search index if it doesn't exist."""
        try:
            if await self.index_exists():
                logger.info(f"Index '{self.index_name}' already exists.")
                return False
            
            schema = self.get_index_schema()
            await self.index_client.create_index(schema)
            logger.info(f"Created index '{self.index_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise ProviderException(f"Failed to create index: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def index_exists(self) -> bool:
        """Check if an index exists."""
        try:
            await self.index_client.get_index(self.index_name)
            return True
        except Exception:
            return False

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_index(self) -> bool:
        """Delete the search index."""
        try:
            await self.index_client.delete_index(self.index_name)
            logger.info(f"Deleted index '{self.index_name}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            raise ProviderException(f"Failed to delete index: {e}")
