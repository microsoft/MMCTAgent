"""Azure AI Search provider implementation for temporal events index."""

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
from mmct.providers.base.temporal_vector_db_provider import BaseTemporalVectorDBProvider
from mmct.providers.search_document_models import TemporalEventIndexDocument
from mmct.providers.azure_providers.azure_schema_utils import (
    create_azure_index_schema,
    parse_azure_response_to_model,
)


class AISearchTemporalProvider(BaseTemporalVectorDBProvider):
    """Azure AI Search provider implementation for temporal events."""

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
        """Creates Azure AI Search specific schema based on TemporalEventIndexDocument type."""
        return create_azure_index_schema(
            model_class=TemporalEventIndexDocument,
            index_name=self.index_name,
            vector_dimensions=self.dimensions or 1536,
            vector_field_name="description_vector",
        )

    def parse_response(self, vector_db_document: Any) -> TemporalEventIndexDocument:
        """Parses the retrieved Azure vector DB document into TemporalEventIndexDocument object."""
        return parse_azure_response_to_model(vector_db_document, TemporalEventIndexDocument)

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def upload_events(self, events: List[TemporalEventIndexDocument]) -> bool:
        """Batch upload temporal events."""
        try:
            if not events:
                logger.warning("No events to upload")
                return True
            
            client = self._get_search_client()
            documents = [event.model_dump() for event in events]
            
            result = await client.upload_documents(documents=documents)
            
            success_count = sum(1 for r in result if r.succeeded)
            total_count = len(result)
            
            logger.info(f"Uploaded {success_count}/{total_count} temporal events")
            
            return success_count == total_count
        except Exception as e:
            logger.error(f"Failed to upload events: {e}")
            raise ProviderException(f"Failed to upload events: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search_by_time_range(
        self,
        video_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[TemporalEventIndexDocument]:
        """Search events within a time range."""
        try:
            client = self._get_search_client()
            
            # Build OData filter
            filters = [f"video_id eq '{video_id}'"]
            
            if start_time is not None:
                filters.append(f"start_timestamp ge {start_time}")
            
            if end_time is not None:
                filters.append(f"end_timestamp le {end_time}")
            
            if event_type is not None:
                filters.append(f"event_type eq '{event_type}'")
            
            filter_str = " and ".join(filters)
            
            results = await client.search(
                search_text="*",
                filter=filter_str,
                order_by="start_timestamp asc",
                top=limit,
            )
            
            parsed_results = []
            async for result in results:
                document = self.parse_response(dict(result))
                parsed_results.append(document)
            
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to search by time range: {e}")
            raise ProviderException(f"Failed to search by time range: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def get_events_in_sequence(
        self,
        video_id: str,
        start_sequence: int,
        end_sequence: int,
    ) -> List[TemporalEventIndexDocument]:
        """Get events by sequence number range - optimized for chain traversal."""
        try:
            client = self._get_search_client()
            
            filter_str = (
                f"video_id eq '{video_id}' and "
                f"sequence_number ge {start_sequence} and "
                f"sequence_number le {end_sequence}"
            )
            
            results = await client.search(
                search_text="*",
                filter=filter_str,
                order_by="sequence_number asc",
                top=end_sequence - start_sequence + 1,
            )
            
            parsed_results = []
            async for result in results:
                document = self.parse_response(dict(result))
                parsed_results.append(document)
            
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to get events in sequence: {e}")
            raise ProviderException(f"Failed to get events in sequence: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def get_events_by_relationship(
        self,
        target_event_id: str,
        relationship: str,
        limit: int = 10,
    ) -> List[TemporalEventIndexDocument]:
        """Get events by relationship using indexed filters - O(1) lookup."""
        try:
            client = self._get_search_client()
            
            # Build filter based on relationship type
            # Uses Azure AI Search collection/any syntax for filtering arrays
            if relationship == "precedes":
                # Find events where precedes_event_ids contains target_event_id
                filter_str = f"precedes_event_ids/any(id: id eq '{target_event_id}')"
            elif relationship == "follows":
                # Find events where follows_event_ids contains target_event_id
                filter_str = f"follows_event_ids/any(id: id eq '{target_event_id}')"
            else:
                raise ValueError(f"Invalid relationship type: {relationship}. Must be 'precedes' or 'follows'")
            
            results = await client.search(
                search_text="*",
                filter=filter_str,
                order_by="sequence_number asc",
                top=limit,
            )
            
            parsed_results = []
            async for result in results:
                document = self.parse_response(dict(result))
                parsed_results.append(document)
            
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to get events by relationship: {e}")
            raise ProviderException(f"Failed to get events by relationship: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def semantic_search_events(
        self,
        query: str,
        query_vector: List[float],
        video_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[TemporalEventIndexDocument]:
        """Semantic search across event descriptions."""
        try:
            client = self._get_search_client()
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="description_vector"
            )
            
            # Apply video filter if provided
            filter_str = f"video_id eq '{video_id}'" if video_id else None
            
            results = await client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=filter_str,
                top=top_k,
            )
            
            parsed_results = []
            async for result in results:
                document = self.parse_response(dict(result))
                parsed_results.append(document)
            
            return parsed_results
        except Exception as e:
            logger.error(f"Failed to semantic search events: {e}")
            raise ProviderException(f"Failed to semantic search events: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_events(self, video_id: str) -> bool:
        """Delete all events for a video."""
        try:
            client = self._get_search_client()
            
            # First, find all events for this video
            results = await client.search(
                search_text="*",
                filter=f"video_id eq '{video_id}'",
                select=["event_id"],
                top=10000,  # Large number to get all events
            )
            
            event_ids = []
            async for result in results:
                event_ids.append({"event_id": result["event_id"]})
            
            if not event_ids:
                logger.info(f"No events found for video: {video_id}")
                return True
            
            # Delete all found events
            delete_result = await client.delete_documents(documents=event_ids)
            
            success_count = sum(1 for r in delete_result if r.succeeded)
            logger.info(f"Deleted {success_count}/{len(event_ids)} events for video: {video_id}")
            
            return success_count == len(event_ids)
        except Exception as e:
            logger.error(f"Failed to delete events: {e}")
            raise ProviderException(f"Failed to delete events: {e}")

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
