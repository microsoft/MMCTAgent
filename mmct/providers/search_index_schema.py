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


def create_video_chapter_index_schema(index_name: str, dimensions: int = 1536) -> SearchIndex:
    """
    Create the index schema definition for video chapter search.
    This schema is based on AISearchDocument model.

    Args:
        index_name: Name of the index to create
        dimensions: Dimensionality of the embedding vectors (default: 1536)

    Returns:
        SearchIndex: The index schema definition
    """
    from mmct.providers.search_document_models import ChapterIndexDocument

    # Create index definition using AISearchDocument model fields
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
                    vector_search_dimensions=dimensions,  # Configurable dimension from provider
                    vector_search_profile_name="embedding_profile"
                )
            )
            continue

        # Choose data type based on annotation
        if model_field.annotation is datetime:
            data_type = SearchFieldDataType.DateTimeOffset
        elif model_field.annotation is float:
            data_type = SearchFieldDataType.Double
        elif model_field.annotation is int:
            data_type = SearchFieldDataType.Int32
        else:
            data_type = SearchFieldDataType.String

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

    print(fields)
    return index


def create_object_collection_index_schema(index_name: str, dimensions: int = 1536) -> SearchIndex:
    """
    Create the index schema definition for object collection search.
    This schema is based on ObjectCollectionDocument model.

    Args:
        index_name: Name of the index to create
        dimensions: Dimensionality of the embedding vectors (default: 1536)

    Returns:
        SearchIndex: The index schema definition
    """
    from mmct.providers.search_document_models import ObjectCollectionDocument

    # Create index definition using ObjectCollectionDocument model fields
    fields = []

    for name, model_field in ObjectCollectionDocument.model_fields.items():
        extra = model_field.json_schema_extra

        # Special handling for embeddings vector field
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
                    vector_search_dimensions=dimensions,  # Configurable dimension from provider
                    vector_search_profile_name="embedding_profile"
                )
            )
            continue

        # Determine data type based on annotation
        if model_field.annotation is float:
            data_type = SearchFieldDataType.Double
        elif model_field.annotation is int:
            data_type = SearchFieldDataType.Int32
        else:
            data_type = SearchFieldDataType.String

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
            fields.append(
                SearchableField(
                    **common_kwargs,
                    analyzer_name="en.microsoft"
                )
            )
        else:
            fields.append(
                SimpleField(**common_kwargs)
            )
    important_fields = [
        SemanticField(field_name="video_summary")
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

    # Create the index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_config

    )
    return index


def create_keyframe_index_schema(index_name: str, dimensions: int = 512) -> SearchIndex:
    """
    Create Azure AI Search index schema for keyframes.

    Args:
        index_name: Name of the index to create
        dimensions: Dimensionality of the CLIP embedding vectors (default: 512)

    Returns:
        SearchIndex: Azure-specific index schema definition
    """
    fields = [
        # identifier
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        # metadata fields
        SearchableField(
            name="video_id", type=SearchFieldDataType.String, filterable=True, facetable=True
        ),
        SearchableField(
            name="keyframe_filename",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # vector field for CLIP embeddings
        SearchField(
            name="embeddings",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,  # Configurable dimension from provider
            vector_search_profile_name="clip-profile",
        ),
        SimpleField(
            name="created_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="motion_score", type=SearchFieldDataType.Double, filterable=True, sortable=True
        ),
        SimpleField(
            name="timestamp_seconds",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(name="blob_url", type=SearchFieldDataType.String),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(
            name="parent_duration",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="video_duration",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
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
            VectorSearchProfile(
                name="clip-profile", algorithm_configuration_name="hnsw-algorithm"
            )
        ],
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    return index
