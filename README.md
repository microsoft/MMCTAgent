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

### 5. Test MCP Server (Optional)

Test the server and list available tools:

```bash
python client.py
```

### 6. Use Search Tool

Connect your MCP client to use the `get_visual_timestamps` tool for searching video frames.

## Requirements

- Python 3.8+
- Azure AI Search and Blob Storage
- Azure CLI (`az login` before ingestion)
