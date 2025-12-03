"""
Helper utilities for Azure AI Search schema generation from Pydantic models.
"""

from typing import Type, List
from pydantic import BaseModel
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)


def pydantic_field_to_azure_field(
    field_name: str,
    field_info,
    field_type
) -> SearchField:
    """
    Convert a Pydantic Field to an Azure SearchField.

    Args:
        field_name: Name of the field
        field_info: Pydantic FieldInfo object
        field_type: Python type of the field

    Returns:
        SearchField: Azure SearchField object
    """
    # Extract metadata from Pydantic Field's json_schema_extra
    metadata = field_info.json_schema_extra if field_info.json_schema_extra else {}

    # Get field properties
    is_key = metadata.get("key", False)
    searchable = metadata.get("searchable", False)
    filterable = metadata.get("filterable", False)
    sortable = metadata.get("sortable", False)
    facetable = metadata.get("facetable", False)
    retrievable = metadata.get("retrievable", True)

    # Map Python types to Azure SearchFieldDataType
    type_mapping = {
        str: SearchFieldDataType.String,
        int: SearchFieldDataType.Int32,
        float: SearchFieldDataType.Double,
        bool: SearchFieldDataType.Boolean,
        "datetime": SearchFieldDataType.DateTimeOffset,
        "list_float": SearchFieldDataType.Collection(SearchFieldDataType.Single),
    }

    # Determine Azure field type
    if field_type == "datetime" or str(field_type) == "<class 'datetime.datetime'>":
        azure_type = SearchFieldDataType.DateTimeOffset
    elif hasattr(field_type, "__origin__") and field_type.__origin__ == list:
        # Handle List[float] for embeddings
        azure_type = SearchFieldDataType.Collection(SearchFieldDataType.Single)
    else:
        azure_type = type_mapping.get(field_type, SearchFieldDataType.String)

    # Create appropriate field based on properties
    if is_key:
        return SimpleField(
            name=field_name,
            type=azure_type,
            key=True,
            filterable=filterable,
            sortable=sortable,
        )
    elif searchable and azure_type == SearchFieldDataType.String:
        return SearchableField(
            name=field_name,
            type=azure_type,
            searchable=True,
            filterable=filterable,
            sortable=sortable,
            facetable=facetable,
        )
    else:
        return SimpleField(
            name=field_name,
            type=azure_type,
            filterable=filterable,
            sortable=sortable,
            facetable=facetable,
        )


def create_azure_index_schema(
    model_class: Type[BaseModel],
    index_name: str,
    vector_dimensions: int = 1536,
    vector_field_name: str = "embeddings",
) -> SearchIndex:
    """
    Generate an Azure SearchIndex schema from a Pydantic model.

    Args:
        model_class: Pydantic model class (e.g., ChapterIndexDocument)
        index_name: Name of the index
        vector_dimensions: Dimension of vector embeddings
        vector_field_name: Name of the vector field

    Returns:
        SearchIndex: Azure SearchIndex object
    """
    fields = []

    # Introspect Pydantic model fields
    for field_name, field_info in model_class.model_fields.items():
        field_type = field_info.annotation

        # Handle vector field specially
        if field_name == vector_field_name:
            fields.append(
                SearchField(
                    name=field_name,
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=vector_dimensions,
                    vector_search_profile_name="vector-profile",
                )
            )
        else:
            # Handle datetime specially
            if "datetime" in str(field_type).lower():
                fields.append(
                    pydantic_field_to_azure_field(field_name, field_info, "datetime")
                )
            else:
                fields.append(
                    pydantic_field_to_azure_field(field_name, field_info, field_type)
                )

    # Configure vector search
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
                name="vector-profile",
                algorithm_configuration_name="hnsw-algorithm",
            )
        ],
    )

    # Create semantic configuration (if applicable)
    semantic_config = SemanticConfiguration(
        name="my-semantic-search-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=None,
            content_fields=[
                SemanticField(field_name=field_name)
                for field_name, field_info in model_class.model_fields.items()
                if field_info.json_schema_extra
                and field_info.json_schema_extra.get("searchable", False)
                and field_info.annotation == str
            ],
        ),
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    return SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def parse_azure_response_to_model(
    azure_document: dict,
    model_class: Type[BaseModel]
) -> BaseModel:
    """
    Parse Azure search response document to Pydantic model.

    Args:
        azure_document: Dictionary from Azure search results
        model_class: Target Pydantic model class

    Returns:
        Instance of model_class
    """
    # Remove Azure-specific fields
    clean_doc = {
        k: v for k, v in azure_document.items()
        if not k.startswith("@search")
    }

    # Create and return model instance
    return model_class(**clean_doc)


def extract_similarity_score(azure_document: dict) -> float:
    """
    Extract similarity score from Azure search response.

    Args:
        azure_document: Dictionary from Azure search results

    Returns:
        Similarity score (defaults to 0.0 if not found)
    """
    return azure_document.get("@search.score", 0.0)
