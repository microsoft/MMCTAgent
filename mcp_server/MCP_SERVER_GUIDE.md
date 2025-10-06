# MCP Server Guide

This is a FastMCP server that exposes video keyframe search as an MCP tool.

## Overview

The MCP server provides a `get_visual_timestamps` tool that can be called by MCP clients (like Claude Desktop, IDEs, etc.) to search video keyframes using natural language queries.

## Architecture

```
MCP Client (Claude, IDE, etc.)
    ↓
    MCP Protocol (HTTP/SSE)
    ↓
FastMCP Server (port 8000)
    ↓
get_visual_timestamps tool
    ↓
Azure AI Search (CLIP embeddings)
```

## Installation

```bash
cd mcp_server
pip install -r requirements.txt
```

## Configuration

The server reads configuration from `/home/v-amanpatkar/work/lively/.env`:

```bash
# Required
SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
SEARCH_INDEX_NAME=test_lively

# Optional (use Azure CLI authentication if not provided)
SEARCH_API_KEY=your-search-api-key
```

## Running the Server

### Method 1: Direct Run
```bash
cd /home/v-amanpatkar/work/lively
python -m mcp_server.main
```

### Method 2: From mcp_server directory
```bash
cd mcp_server
python main.py
```

The server will start on:
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 8000
- **Transport**: streamable-http

## MCP Tool: get_visual_timestamps

### Description
Searches video keyframes using CLIP embeddings and returns matching timestamps.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Natural language description of what to find |
| `top_k` | integer | No | 3 | Number of results to return |
| `video_id` | string | No | null | Filter results by specific video |

### Returns
Comma-separated string of timestamps in seconds.

**Example**: `"5.0, 12.5, 18.0"`

### Example Usage

#### From MCP Client (e.g., Claude Desktop)

```
User: Find moments where people are walking
Claude: [calls get_visual_timestamps with query="person walking"]
Tool Response: "5.0, 12.5, 18.0"
Claude: I found people walking at timestamps 5.0s, 12.5s, and 18.0s
```

#### Direct API Call (for testing)

```bash
curl -X POST http://localhost:8000/tools/get_visual_timestamps \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person walking",
    "top_k": 3
  }'
```

## Testing the Server

### 1. Start the Server
```bash
python -m mcp_server.main
```

You should see:
```
[INFO] Loaded environment from: /home/v-amanpatkar/work/lively/.env
[INFO] Starting MCP Server...
[INFO] Search Endpoint: https://osaistemp.search.windows.net
[INFO] Index Name: test_lively
[INFO] Server running on http://0.0.0.0:8000
```

### 2. Test the Tool

**Option A: Using MCP Inspector**
```bash
npx @modelcontextprotocol/inspector http://localhost:8000
```

**Option B: Using curl**
```bash
curl http://localhost:8000/tools
```

**Option C: Call the tool directly**
```python
import requests

response = requests.post(
    "http://localhost:8000/tools/get_visual_timestamps",
    json={
        "query": "person walking",
        "top_k": 3
    }
)

print(response.json())
# Output: {"result": "5.0, 12.5, 18.0"}
```

## Connecting to MCP Clients

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "lively-visual-search": {
      "url": "http://localhost:8000"
    }
  }
}
```

### Other MCP Clients

Configure with:
- **Transport**: HTTP
- **URL**: `http://localhost:8000`
- **Protocol**: MCP

## Query Examples

### People & Actions
```
"person walking"
"people talking"
"someone running"
"handshake"
```

### Vehicles
```
"car driving"
"bicycle"
"traffic scene"
```

### Nature
```
"sunset"
"ocean waves"
"mountain landscape"
```

### Urban
```
"cityscape"
"street scene"
"building exterior"
```

## Error Handling

The tool returns error messages as strings:

```python
# Missing configuration
"Error: SEARCH_SERVICE_ENDPOINT not configured in environment"

# No results found
"No keyframes found for query: 'flying unicorn'"

# Search error
"Error searching keyframes: Connection timeout"
```

## Development

### Project Structure
```
mcp_server/
├── server.py              # FastMCP instance
├── main.py               # Entry point
├── tools/
│   ├── get_visual_timestamps.py    # MCP tool
│   └── visual_search_client.py     # Search logic
├── requirements.txt
└── MCP_SERVER_GUIDE.md   # This file
```

### Adding New Tools

1. Create tool function in `tools/`
2. Decorate with `@mcp.tool()`
3. Import in `main.py`
4. Tool is automatically registered

Example:
```python
from ..server import mcp

@mcp.tool(
    name="my_tool",
    description="Description of what this tool does"
)
def my_tool(param: str) -> str:
    return f"Result: {param}"
```

## Troubleshooting

### Server won't start

**Check environment:**
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print(os.getenv('SEARCH_SERVICE_ENDPOINT'))"
```

**Check dependencies:**
```bash
pip install -r requirements.txt
```

### Tool returns errors

**"SEARCH_SERVICE_ENDPOINT not configured"**
- Verify `.env` file exists at `lively/.env`
- Check environment variable is set correctly

**"No keyframes found"**
- Run ingestion first: `python run_example.py`
- Verify data in Azure AI Search
- Try different query phrasing

**Connection errors**
- Run `az login` for Azure CLI authentication
- Or provide `SEARCH_API_KEY` in `.env`

### Tool returns empty results

- Check if videos have been ingested
- Verify index name matches: `SEARCH_INDEX_NAME=test_lively`
- Try more general queries

## Production Deployment

### Security Considerations

1. **Don't expose to internet** without authentication
2. **Use HTTPS** in production
3. **Set API keys** securely (Azure Key Vault, etc.)
4. **Rate limiting** if public-facing

### Deployment Options

**Docker:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY mcp_server/ ./mcp_server/
COPY .env ./
RUN pip install -r mcp_server/requirements.txt
CMD ["python", "-m", "mcp_server.main"]
```

**systemd service:**
```ini
[Unit]
Description=Lively MCP Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/v-amanpatkar/work/lively
ExecStart=/usr/bin/python3 -m mcp_server.main
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance

- **First request**: ~2-3s (CLIP model loading)
- **Subsequent requests**: ~200-500ms per query
- **Concurrent requests**: Supported (async)
- **Memory**: ~2GB (CLIP model)

## Monitoring

### Logs
The server uses `loguru` for logging. Logs include:
- Server startup
- Environment configuration
- Tool calls
- Errors

### Metrics
Monitor:
- Request latency
- Error rate
- CLIP model load time
- Azure Search response time

## Support

- Main docs: `/home/v-amanpatkar/work/lively/README.md`
- Query examples: `/home/v-amanpatkar/work/lively/QUERY_EXAMPLES.md`
- MCP Protocol: https://modelcontextprotocol.io/
- FastMCP docs: https://github.com/jlowin/fastmcp
