# Lively - Video Frame Search with MCP

Video ingestion and search system that extracts frames, stores them in Azure Blob Storage and AI Search, and provides semantic search through an MCP server.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_STORAGE_ACCOUNT_URL=https://your-storage-account.blob.core.windows.net
SEARCH_INDEX_NAME=<some-name>
```

### 3. Ingest Video

Update video path in `run_ingestion.py`, then run:

```bash
python run_ingestion.py
```

Extracts frames and stores them in Azure Blob Storage and AI Search.

### 4. Start MCP Server

```bash
python -m mcp_server.main
```

### 5. Query Video Frames

Use the client to search for video frames using natural language queries:

```bash
# Basic search
python client.py --query 'person walking'

# Search with more results
python client.py --query 'car driving' --top-k 5

# Filter by video ID
python client.py --query 'person walking' --video-id video123

# Filter by YouTube URL
python client.py --query 'sunset scene' --youtube-url 'https://youtube.com/watch?v=xyz'

# Use custom server URL
python client.py --query 'person walking' --server-url 'http://localhost:8000/mcp'
```

**Available Options:**
- `--query` (required): Text description of what to search for
- `--top-k` (optional, default: 3): Number of top results to return
- `--video-id` (optional): Filter results by video ID
- `--youtube-url` (optional): Filter results by YouTube URL (takes precedence over video-id)
- `--server-url` (optional, default: http://0.0.0.0:8000/mcp): MCP server URL

### 6. Use Search Tool in MCP Clients

Connect any MCP client to use the `get_visual_timestamps` tool for searching video frames programmatically.

## Requirements

- Python 3.8+
- Azure AI Search and Blob Storage
- Azure CLI (`az login` before ingestion)
