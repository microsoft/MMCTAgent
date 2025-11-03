
from mmct.utils.error_handler import handle_exceptions, convert_exceptions
from mmct.utils.error_handler import ProviderException, ConfigurationException
from loguru import logger
from typing import Dict, Any, List, Optional
from datetime import datetime
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    ExhaustiveKnnAlgorithmConfiguration,
    VectorSearchProfile
)
from mmct.providers.base import SearchProvider
from mmct.providers.credentials import AzureCredentials

class AzureSearchProvider(SearchProvider):
    """Azure AI Search provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.credential = AzureCredentials.get_async_credentials()
        self.index_client = self._initialize_index_client()
        # Cache for search clients with different index names
        self._client_cache: Dict[str, SearchClient] = {}
        # Store the default index name for convenience
        self._default_index_name = self.config.get("index_name", "default")
    
    def _create_search_client(self, index_name: str) -> SearchClient:
        """
        Create a SearchClient for a specific index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            SearchClient instance
        """
        try:
            endpoint = self.config.get("endpoint")
            if not endpoint:
                raise ConfigurationException("SEARCH_ENDPOINT environment variable not set")

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

    def _initialize_index_client(self) -> SearchIndexClient:
        """Initialize Azure AI Search Index client for index management."""
        try:
            endpoint = self.config.get("endpoint")
            if not endpoint:
                raise ConfigurationException("Azure AI Search endpoint is required")

            return SearchIndexClient(endpoint=endpoint, credential=self.credential)
        except Exception as e:
            raise ProviderException(f"Failed to initialize Azure AI Search Index client: {e}")
    
    def _get_client_for_index(self, index_name: Optional[str] = None) -> SearchClient:
        """
        Get or create a SearchClient for the specified index.
        Uses caching to avoid creating multiple clients for the same index.
        
        Args:
            index_name: Name of the index. If None, uses the default index.
            
        Returns:
            SearchClient instance for the specified index
        """
        # Use default index name if not specified
        if not index_name:
            index_name = self._default_index_name
        
        # Check cache for existing client
        if index_name in self._client_cache:
            return self._client_cache[index_name]
        
        # Create and cache new client
        new_client = self._create_search_client(index_name)
        self._client_cache[index_name] = new_client
        return new_client

    def _create_video_chapter_index_schema(self, index_name: str, dim: int = 1536) -> SearchIndex:
        """
        Create Azure AI Search index schema for video chapters.

        Args:
            index_name: Name of the index to create
            dim: Dimensionality of the text embedding vectors (default: 1536 for text-embedding-ada-002)

        Returns:
            SearchIndex: Azure-specific index schema definition
        """
        from mmct.providers.search_document_models import ChapterIndexDocument

        # Create index definition using ChapterIndexDocument model fields
        fields = []
        searchable_fields_names = []

        for name, model_field in ChapterIndexDocument.model_fields.items():
            extra = model_field.json_schema_extra

            # Special handling for embeddings vector
            if name == "embeddings":
                fields.append(
                    SearchField(
                        name=name,
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,
                        filterable=extra.get("filterable", False),
                        facetable=extra.get("facetable", False),
                        sortable=extra.get("sortable", False),
                        hidden=not extra.get("stored", True),
                        vector_search_dimensions=dim,
                        vector_search_profile_name="embedding_profile"
                    )
                )
                continue

            # Choose data type
            data_type = (
                SearchFieldDataType.DateTimeOffset
                if model_field.annotation is datetime
                else SearchFieldDataType.String
            )

            common_kwargs = dict(
                name=name,
                type=data_type,
                key=extra.get("key", False),
                filterable=extra.get("filterable", False),
                facetable=extra.get("facetable", False),
                sortable=extra.get("sortable", False),
                retrievable=extra.get("retrievable", True),
                hidden=not extra.get("stored", True),
            )

            if extra.get("searchable", False):
                searchable_fields_names.append(name)
                fields.append(
                    SearchableField(
                        **common_kwargs,
                        analyzer_name="en.microsoft"  # or your preferred analyzer
                    )
                )
            else:
                fields.append(
                    SimpleField(**common_kwargs)
                )

        # Configure semantic search
        important_fields = [
            SemanticField(field_name="chapter_transcript"),
            SemanticField(field_name="text_from_scene"),
            SemanticField(field_name="action_taken"),
            SemanticField(field_name="detailed_summary")
        ]
        semantic_config = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="my-semantic-search-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        content_fields=important_fields,
                        keywords_fields=important_fields
                    )
                )
            ]
        )

        # Configure vector search algorithms
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw_config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="myExhaustiveKnn",
                    parameters={
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="embedding_profile",
                    algorithm_configuration_name="hnsw_config"
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="myExhaustiveKnn"
                )
            ]
        )

        # Create the index with all configurations
        index = SearchIndex(
            name=index_name,
            fields=fields,
            semantic_search=semantic_config,
            vector_search=vector_search
        )
        return index

    def _create_keyframe_index_schema(self, index_name: str, dim: int = 512) -> SearchIndex:
        """
        Create Azure AI Search index schema for keyframes.

        Args:
            index_name: Name of the index to create
            dim: Dimensionality of the CLIP embedding vectors (default: 512)

        Returns:
            SearchIndex: Azure-specific index schema definition
        """
        fields = [
            # identifier
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            # metadata fields
            SearchableField(name="video_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="keyframe_filename", type=SearchFieldDataType.String, filterable=True, facetable=True),
            # vector field for CLIP embeddings
            SearchField(
                name="embeddings",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dim,
                vector_search_profile_name="clip-profile",
            ),
            SimpleField(name="created_at", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SimpleField(name="motion_score", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SimpleField(name="timestamp_seconds", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SimpleField(name="blob_url", type=SearchFieldDataType.String),
            SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="parent_duration", type=SearchFieldDataType.Double, filterable=True, sortable=True),
            SimpleField(name="video_duration", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-algorithm",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine",
                    },
                )
            ],
            profiles=[
                VectorSearchProfile(name="clip-profile", algorithm_configuration_name="hnsw-algorithm")
            ],
        )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        return index

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def search(self, query: str, index_name: str = None, **kwargs) -> List[Dict]:
        """
        Search documents using Azure AI Search.
        
        Args:
            query: Search query string
            index_name: Optional index name (uses default if not provided)
            **kwargs: Additional search parameters including:
                - search_text: Text to search for (defaults to query)
                - top: Number of results to return
                - embedding: Vector embedding for vector search
                - query_type: Type of query (e.g., "semantic", "vector")
                - vector_queries: Pre-built vector queries
                - semantic_configuration_name: Name of semantic configuration
                
        Returns:
            List of matching documents
        """
        try:
            # Extract search parameters
            search_text = kwargs.pop("search_text", query)
            top = kwargs.pop("top", None)
            embedding = kwargs.pop("embedding", [])
            query_type = kwargs.pop("query_type", None)
            vector_queries = kwargs.pop("vector_queries", None)
            semantic_configuration_name = None

            # Handle semantic search configuration
            if query_type == "semantic":
                semantic_configuration_name = kwargs.pop("semantic_configuration_name", "my-semantic-search-config")
                search_text = None
                
            # Handle vector search configuration
            if query_type == "vector":
                query_type = None
                
            # Build vector queries if embedding provided
            if embedding and top and not vector_queries:
                vector_query = VectorizedQuery(
                    vector=embedding, k_nearest_neighbors=top, fields="embeddings"
                )
                vector_queries = [vector_query]

            # Get appropriate client for the index
            client = self._get_client_for_index(index_name)
            
            # Execute search
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
        """
        Index a document in Azure AI Search.
        
        Args:
            document: Document dictionary to index
            index_name: Optional index name (uses default if not provided)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = self._get_client_for_index(index_name)
            result = await client.upload_documents(documents=[document])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search indexing failed: {e}")
            raise ProviderException(f"Azure AI Search indexing failed: {e}")
    
    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def delete_document(self, doc_id: str, index_name: str = None) -> bool:
        """
        Delete a document from Azure AI Search.
        
        Args:
            doc_id: ID of the document to delete
            index_name: Optional index name (uses default if not provided)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            client = self._get_client_for_index(index_name)
            result = await client.delete_documents(documents=[{"id": doc_id}])
            return result[0].succeeded
        except Exception as e:
            logger.error(f"Azure AI Search deletion failed: {e}")
            raise ProviderException(f"Azure AI Search deletion failed: {e}")

    @handle_exceptions(retries=3, exceptions=(Exception,))
    @convert_exceptions({Exception: ProviderException})
    async def create_index(self, index_name: str, index_schema: Any) -> bool:
        """
        Create a search index with the given schema.

        Args:
            index_name: Name of the index to create
            index_schema: Can be:
                - String: "chapter" or "keyframe" for standard schemas
                - SearchIndex: Azure SearchIndex object (used directly)
                - Dict: with "type" and optional params like {"type": "keyframe", "dim": 512} or {"type": "chapter", "dim": 1536}

        Returns:
            bool: True if created, False if already exists
        """
        try:
            # Handle different schema input formats
            if isinstance(index_schema, str):
                # Simple string indicator
                if index_schema == "keyframe":
                    index_schema = self._create_keyframe_index_schema(index_name)
                elif index_schema == "chapter":
                    index_schema = self._create_video_chapter_index_schema(index_name)
                else:
                    raise ProviderException(f"Unknown index schema type: {index_schema}")
            
            elif isinstance(index_schema, dict):
                # Dict with type indicator and optional params
                schema_type = index_schema.get("type", "chapter")
                if schema_type == "keyframe":
                    dim = index_schema.get("dim", 512)
                    index_schema = self._create_keyframe_index_schema(index_name, dim)
                elif schema_type == "chapter":
                    dim = index_schema.get("dim", 1536)
                    index_schema = self._create_video_chapter_index_schema(index_name, dim)
                else:
                    raise ProviderException(f"Unknown index schema type: {schema_type}")
            
            # Otherwise assume it's already a SearchIndex object
            
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
            Dict with upload results including success status, count, and result details
        """
        try:
            client = self._get_client_for_index(index_name)
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
            client = self._get_client_for_index(index_name)

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
        """Close all search clients and cleanup resources."""
        try:
            # Close all cached clients
            for index_name, client in self._client_cache.items():
                logger.info(f"Closing Azure AI Search client for index '{index_name}'")
                await client.close()
            
            # Clear the cache
            self._client_cache.clear()
            
            # Close index client
            if self.index_client:
                logger.info("Closing Azure AI Search Index client")
                await self.index_client.close()
        except Exception as e:
            logger.error(f"Error during client cleanup: {e}")
            # Don't raise exception during cleanup to avoid masking original errors