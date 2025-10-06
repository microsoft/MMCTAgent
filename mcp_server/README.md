# Lively MCP Server

Model Context Protocol (MCP) server for visual search and video understanding using CLIP embeddings and Azure AI Search.

## Overview

This module provides tools for searching video keyframes using natural language queries. It:
1. Embeds text queries using CLIP (the same model used to embed keyframes during ingestion)
2. Performs vector similarity search in Azure AI Search
3. Returns timestamps, filenames, and blob URLs of matching keyframes

## Features

- **Natural Language Search**: Search video keyframes using descriptive text queries
- **CLIP Embeddings**: Uses the same CLIP model as the ingestion pipeline for consistency
- **Vector Search**: Leverages Azure AI Search's vector similarity search
- **Flexible Filtering**: Filter results by video ID
- **Multiple Output Formats**: Get full summaries or just timestamps

## Installation

The MCP server uses the same dependencies as the ingestion pipeline. Ensure you have:

```bash
pip install torch transformers pillow azure-search-documents azure-identity
```

## Configuration

Set the following environment variables (or use `.env` file):

```bash
# Required
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
SEARCH_INDEX_NAME=video-keyframes-index

# Optional (if not using Azure CLI authentication)
SEARCH_API_KEY=your-search-api-key

# Optional model configuration
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
DEFAULT_TOP_K=10
```

## Usage

### Basic Search

```python
from mcp_server.tools import get_visual_summary

# Search for keyframes
summary = get_visual_summary(
    query="person walking on the street",
    search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
    index_name="video-keyframes-index",
    top_k=10
)

print(f"Found {summary['total_results']} keyframes")
for keyframe in summary['keyframes']:
    print(f"  {keyframe['timestamp_seconds']:.2f}s - {keyframe['blob_url']}")
```

### Get Timestamps Only

```python
from mcp_server.tools import get_visual_timestamps

# Get just timestamps
timestamps = get_visual_timestamps(
    query="car driving",
    search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
    index_name="video-keyframes-index",
    top_k=5
)

for ts in timestamps:
    print(f"{ts['timestamp_seconds']:.2f}s - {ts['keyframe_filename']}")
```

### Filter by Video

```python
# Search within a specific video
summary = get_visual_summary(
    query="outdoor scene",
    search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
    index_name="video-keyframes-index",
    video_id="abc123def456",
    top_k=10
)
```

### Using the Class

```python
from mcp_server.tools import VisualSearchTool

# Initialize tool once
tool = VisualSearchTool(
    search_endpoint=os.getenv("SEARCH_SERVICE_ENDPOINT"),
    index_name="video-keyframes-index"
)

# Perform multiple searches
summary1 = tool.get_visual_summary("person walking", top_k=5)
summary2 = tool.get_visual_summary("car driving", top_k=5)
```

## API Reference

### `get_visual_summary()`

Search for keyframes and get a full summary.

**Parameters:**
- `query` (str): Text query describing what to search for
- `search_endpoint` (str): Azure AI Search endpoint URL
- `index_name` (str): Name of the search index
- `search_api_key` (str, optional): API key for Azure Search
- `top_k` (int, default=10): Number of results to return
- `video_id` (str, optional): Filter results by video ID
- `clip_model_name` (str, default="openai/clip-vit-base-patch32"): CLIP model to use

**Returns:**
```python
{
    "query": "person walking",
    "total_results": 10,
    "keyframes": [
        {
            "id": "uuid",
            "video_id": "abc123",
            "keyframe_filename": "abc123_150.jpg",
            "timestamp_seconds": 5.0,
            "motion_score": 12.5,
            "blob_url": "https://...",
            "youtube_url": "https://...",
            "created_at": "2025-10-06T...",
            "search_score": 0.95
        },
        ...
    ]
}
```

### `get_visual_timestamps()`

Get just timestamps without full metadata.

**Parameters:** Same as `get_visual_summary()`

**Returns:**
```python
[
    {
        "timestamp_seconds": 5.0,
        "keyframe_filename": "abc123_150.jpg",
        "video_id": "abc123",
        "blob_url": "https://...",
        "youtube_url": "https://...",
        "search_score": 0.95
    },
    ...
]
```

## Example Queries

Here are some example queries you can try:

- `"person walking on the street"`
- `"car driving on highway"`
- `"sunset over ocean"`
- `"group of people talking"`
- `"indoor office scene"`
- `"outdoor mountain landscape"`
- `"dog playing in park"`
- `"cityscape at night"`

## How It Works

1. **Query Embedding**: The text query is processed through the CLIP text encoder to generate a 512-dimensional embedding vector

2. **Vector Search**: The embedding is sent to Azure AI Search, which performs cosine similarity search against the indexed keyframe embeddings

3. **Result Ranking**: Results are ranked by similarity score and returned with full metadata

4. **Filtering**: Optional video_id filter narrows results to a specific video

## Authentication

The MCP server supports multiple authentication methods (in priority order):

1. **Azure CLI** (recommended): Run `az login` before using
2. **Managed Identity**: Automatically used in Azure environments
3. **API Key**: Provide via `search_api_key` parameter

## Examples

Run the example script to see the tools in action:

```bash
cd mcp_server
python example_usage.py
```

## Integration with Ingestion Pipeline

This module is designed to work seamlessly with the ingestion pipeline:

- Uses the same CLIP model (`openai/clip-vit-base-patch32` by default)
- Queries the same Azure AI Search index
- Returns metadata that matches the ingestion output

## Troubleshooting

**Model loading issues:**
- Ensure you have enough disk space for CLIP model (~1GB)
- Check internet connectivity for first-time model download

**Search failures:**
- Verify `SEARCH_SERVICE_ENDPOINT` is correct
- Check authentication (run `az login` or provide API key)
- Ensure the index exists and has data

**No results:**
- Try different query phrasings
- Increase `top_k` value
- Check if keyframes were properly ingested

## Performance Tips

- Initialize `VisualSearchTool` once and reuse for multiple queries
- Use GPU if available for faster CLIP inference
- Adjust `top_k` based on your needs (lower = faster)

## License

Part of the Lively video ingestion and search system.
