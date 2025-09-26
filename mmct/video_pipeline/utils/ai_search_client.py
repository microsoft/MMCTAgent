from typing import Any, Dict, List, Optional, AsyncIterator, Type
import os
import asyncio
from datetime import datetime

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.identity.aio import AzureCliCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
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
    VectorSearchAlgorithmConfiguration,
    VectorSearchProfile
)
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

from loguru import logger


class AISearchClient:
    """
    A client for Azure AI Search that supports common operations with
    error handling, and proper resource cleanup.
    """

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        credential: TokenCredential = None
    ):
        """
        Initialize a new AISearchClient.

        Args:
            endpoint (str): The Azure AI Search service endpoint URL.
            index_name (str): The name of the search index.
            credential (TokenCredential, optional): The credential to use for authentication.
                Defaults to DefaultAzureCredential with AzureCliCredential fallback.
        """
        self.endpoint = endpoint
        self.index_name = index_name
        self.credential = credential or self._get_credential()
        self._search_client = None
        self._index_client = None

        # Initialize clients if index_name is provided
        self._init_search_client()

    def _get_credential(self):
        """Get credential - start with DefaultAzureCredential."""
        logger.info("Using DefaultAzureCredential for authentication")
        return DefaultAzureCredential()

    async def _retry_with_cli_credential(self, operation_func, *args, **kwargs):
        """Retry an operation with AzureCliCredential if DefaultAzureCredential fails."""
        try:
            # First attempt with current credential
            return await operation_func(*args, **kwargs)
        except Exception as e:
            if "Forbidden" in str(e) or "Authentication" in str(e):
                logger.warning(f"Authentication failed with DefaultAzureCredential: {e}")
                logger.info("Retrying with AzureCliCredential...")

                # Switch to AzureCliCredential
                self.credential = AzureCliCredential()

                # Reinitialize clients with new credential
                self._search_client = None
                self._index_client = None
                self._init_search_client()
                self._init_index_client()

                # Retry the operation
                return await operation_func(*args, **kwargs)
            else:
                # Re-raise non-authentication errors
                raise
    
    def _init_search_client(self):
        """Initialize the search client if not already initialized."""
        if not self._search_client and self.index_name:
            self._search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
    
    def _init_index_client(self):
        """Initialize the index client if not already initialized."""
        if not self._index_client:
            self._index_client = SearchIndexClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
    
    @property
    def search_client(self) -> SearchClient:
        """Get the search client, initializing if necessary."""
        self._init_search_client()
        return self._search_client
    
    @property
    def index_client(self) -> SearchIndexClient:
        """Get the index client, initializing if necessary."""
        self._init_index_client()
        return self._index_client
    
    async def close(self):
        """Close all clients to release resources."""
        if self._search_client:
            await self._search_client.close()
            self._search_client = None
        
        if self._index_client:
            await self._index_client.close()
            self._index_client = None
    
    async def upload_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upload documents to the index.
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to upload
            
        Returns:
            Dict[str, Any]: The upload result
        """
        return await self.search_client.upload_documents(documents=documents)
    
    async def check_if_exists(self, hash_id: str) -> bool:
        """
        Check if a document with the given hash_id exists in the index.
        
        Args:
            hash_id (str): The hash_id to check
            
        Returns:
            bool: True if the document exists, False otherwise
        """
        results = await self.search_client.search(
            search_text="*",
            filter=f"hash_video_id eq '{hash_id}'",
            top=1
        )
        docs = [doc async for doc in results]
        return len(docs) > 0
    
    async def check_and_create_index(self) -> bool:
        """
        Create the index if it does not exist.
        
        Returns:
            bool: True if the index was created, False if it already existed
        """
        if not self.index_name:
            raise ValueError("Index name must be provided when creating an index")
        
        self._init_index_client()
        
        # Check if index exists
        try:
            existing_index = await self.index_client.get_index(name=self.index_name)
            logger.info(f"Index '{self.index_name}' already exists")
            if existing_index:
                return False
        except Exception:
            logger.info(f"Index '{self.index_name}' does not exist, will create")
        
        # Create index definition using the index_creator module
        fields = []
        searchable_fields_names = []
        for name, model_field in AISearchDocument.model_fields.items():
            extra = model_field.json_schema_extra
            # special handling for your embeddings vector
            if name == "embeddings":
                fields.append(
                    SearchField(
                        name=name,
                        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True,                  # your annotation
                        filterable=extra.get("filterable", False),
                        facetable=extra.get("facetable", False),
                        sortable=extra.get("sortable", False),
                        hidden=not extra.get("stored", True),
                        vector_search_dimensions=1536,    # e.g. 1536 for text-embedding-ada-002
                        vector_search_profile_name="embedding_profile"
                    )
                )
                continue

            # choose data type
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
        logger.info(f"Creating index '{self.index_name}'...")
        index = SearchIndex(
            name=self.index_name, 
            fields=fields,
            semantic_search=semantic_config,
            vector_search=vector_search
        )
        created = await self.index_client.create_index(index)
        logger.info(f"Successfully created index '{created.name}'")
        
        # Initialize the search client now that the index exists
        self._search_client = None  # Reset so it gets re-initialized with the new index
        self._init_search_client()
        
        return True

class AISearchDocument(BaseModel):
    # — Primary key —
    id: str = Field(
        ...,
        description="Unique document ID",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=True
    )

    # — Searchable text fields —
    topic_of_video: str = Field(
        ...,
        description="What the video is about",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    detailed_summary: str = Field(
        ...,
        description="Long-form summary of the video",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    action_taken: str = Field(
        ...,
        description="Actions described in the video",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    text_from_scene: str = Field(
        ...,
        description="On-screen text detected",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    chapter_transcript: str = Field(
        ...,
        description="Full transcript of the chapter",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    category: str = Field(
        ...,
        description="Primary category",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    sub_category: str = Field(
        ...,
        description="Sub-category",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    subject: str = Field(
        ...,
        description="Main subject or item mentioned",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    variety: str = Field(
        ...,
        description="Variety or type of subject",
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=True,
        key=False
    )
    hash_video_id: str = Field(
        ...,
        searchable=True,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    # — Non-searchable metadata —
    youtube_url: str = Field(
        ...,
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_video_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_audio_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_transcript_file_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_transcript_and_summary_file_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_timestamps_file_url: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )
    blob_frames_folder_path: str = Field(
        ...,
        searchable=False,
        filterable=False,
        retrievable=True,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )

    # — Date & vector fields —
    time: datetime = Field(
        ...,
        description="Ingestion timestamp",
        searchable=False,
        filterable=True,
        retrievable=True,
        stored=True,
        sortable=True,
        facetable=False,
        key=False
    )
    embeddings: List[float] = Field(
        ...,
        description="Vector embedding for semantic search",
        searchable=True,
        filterable=False,
        retrievable=False,
        stored=True,
        sortable=False,
        facetable=False,
        key=False
    )