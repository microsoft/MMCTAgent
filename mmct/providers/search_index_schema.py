"""
Search Index Schema Utility

Provides reusable schema creation for Azure AI Search indices.
"""

from datetime import datetime
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


def create_video_chapter_index_schema(index_name: str) -> SearchIndex:
    """
    Create the index schema definition for video chapter search.
    This schema is based on AISearchDocument model.

    Args:
        index_name: Name of the index to create

    Returns:
        SearchIndex: The index schema definition
    """
    from mmct.providers.search_document_models import AISearchDocument

    # Create index definition using AISearchDocument model fields
    fields = []
    searchable_fields_names = []

    for name, model_field in AISearchDocument.model_fields.items():
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
                    vector_search_dimensions=1536,  # e.g. 1536 for text-embedding-ada-002
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


def create_keyframe_index_schema(index_name: str, dim: int = 512) -> SearchIndex:
    """
    Create the index schema definition for keyframe search. Uses a vector field for CLIP embeddings.

    Args:
        index_name: Name of the index to create
        dim: Dimensionality of the CLIP embedding vectors (default: 512)

    Returns:
        SearchIndex: The index schema definition
    """
    fields = [
        # identifier
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        # metadata fields
        SearchableField(name="video_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="keyframe_filename", type=SearchFieldDataType.String, filterable=True, facetable=True),
        # vector field for CLIP embeddings
        SearchField(
            name="clip_embedding",
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
