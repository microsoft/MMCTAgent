# MMCT Agent MCP Server — Guide & Guidelines

## Overview

The **MMCT Agent MCP Server** exposes the Multi-Modal Critical Thinking (MMCT) framework as a set of tools via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/). It enables AI agents and clients to perform video question-answering, image analysis, video ingestion, frame-level querying, and semantic search — all over HTTP.

Built with [FastMCP](https://gofastmcp.com), the server runs on `http://0.0.0.0:8000/mcp` using the Streamable HTTP transport.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  MCP Client                     │
│          (fastmcp.Client / any MCP client)      │
└────────────────────┬────────────────────────────┘
                     │ HTTP (Streamable-HTTP)
                     ▼
┌─────────────────────────────────────────────────┐
│            FastMCP Server (:8000/mcp)           │
│               mcp_server/main.py                │
├─────────────────────────────────────────────────┤
│  Tools Layer (mcp_server/tools/)                │
│  ┌──────────────┐ ┌──────────────────────────┐  │
│  │video_agent   │ │image_agent               │  │
│  │video_ingest  │ │get_context               │  │
│  │query_frame   │ │get_relevant_frames       │  │
│  │get_object_   │ │get_video_summary         │  │
│  │collection    │ │                          │  │
│  └──────────────┘ └──────────────────────────┘  │
├─────────────────────────────────────────────────┤
│  Config Layer (mcp_server/config.py)            │
│  Provider singletons loaded from .env           │
├─────────────────────────────────────────────────┤
│  MMCT Core (mmct/)                              │
│  VideoAgent, ImageAgent, IngestionPipeline, ... │
└─────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Conda environment with `mmct` installed (`pip install .[all]`)
- Azure credentials configured (CLI login or Managed Identity)
- `.env` file with required environment variables

### 1. Configure Environment

Create a `.env` file in the project root with the following variables:

```env
# LLM (Azure OpenAI)
llm_endpoint=https://<your-endpoint>.openai.azure.com/
llm_deployment_name=<deployment-name>
llm_model_name=<model-name>
llm_api_version=<api-version>

# Embedding Service
embedding_service_endpoint=https://<your-endpoint>.openai.azure.com/
embedding_service_deployment_name=<embedding-deployment>
embedding_service_api_version=<api-version>

# AI Search
search_endpoint=https://<your-search>.search.windows.net
chapter_index_name=<chapter-index>
keyframes_index_name=<keyframes-index>
object_collection_index_name=<object-collection-index>

# Azure Blob Storage
storage_account_name=<storage-account>
keyframe_container_name=<container-name>

# Speech Service (for ingestion)
speech_service_resource_id=<resource-id>
speech_service_region=<region>
```

### 2. Start the Server

```bash
cd /path/to/MMCTAgent
python -m mcp_server.main
```

The server starts at `http://0.0.0.0:8000/mcp`.

A health check endpoint is available at `GET http://0.0.0.0:8000/`.

### 3. Test with the Client

```bash
python -m mcp_server.client
```

Edit the `tools_to_validate` list at the bottom of `client.py` to select which tools to test.

---

## Docker Deployment

### Build Images

```bash
# Build the base image (includes system deps + Python packages)
docker build -t <your-registry>/mmct_base:<tag> . -f Dockerfile.base

# Build the MCP server image (adds source code + .env)
# Pass the base image name via --build-arg
docker build \
  --build-arg BASE_IMAGE=<your-registry>/mmct_base:<tag> \
  -t <your-registry>/mmct_mcp:<tag> \
  . -f mcp_server/Dockerfile.mcp
```

### Run Container

```bash
docker run -d --name mmct_mcp -p 8000:8000 \
  <your-registry>/mmct_mcp:<tag>
```

### Check Logs

```bash
docker logs mmct_mcp
```

---

## Available Tools

### 1. `video_agent_tool` — End-to-End Video QA

The primary tool for answering questions about ingested videos. Internally orchestrates transcript retrieval, frame analysis, and multi-step reasoning with an optional critic agent.

| Parameter          | Type    | Required             | Description                          |
| ------------------ | ------- | -------------------- | ------------------------------------ |
| `query`            | string  | ✅                   | Natural language question            |
| `video_id`         | string  | ❌                   | Constrain search to a specific video |
| `url`              | string  | ❌                   | Video source URL                     |
| `use_critic_agent` | boolean | ❌ (default: `true`) | Enable critic validation             |

**Example:**

```json
{
  "query": "What is this video about?",
  "url": "www.youtube.com/watch?v=2W3BKOSg958",
  "use_critic_agent": true
}
```

---

### 2. `image_agent_tool` — Image Analysis & QA

Analyzes images using vision models (VIT, OCR, recognition, object detection) with optional critic feedback.

| Parameter             | Type         | Required              | Description                                               |
| --------------------- | ------------ | --------------------- | --------------------------------------------------------- |
| `image_url`           | string       | ✅                    | Publicly accessible image URL                             |
| `query`               | string       | ✅                    | Question about the image                                  |
| `use_critic_agent`    | boolean      | ✅                    | Enable critic validation                                  |
| `tools`               | list[string] | ✅                    | Analysis tools: `vit`, `ocr`, `recog`, `object_detection` |
| `stream`              | boolean      | ❌ (default: `false`) | Stream intermediate steps                                 |
| `disable_console_log` | boolean      | ❌ (default: `false`) | Suppress console logs                                     |

**Example:**

```json
{
  "query": "What text is written on the board?",
  "image_url": "https://example.com/image.png",
  "tools": ["ocr", "vit"],
  "use_critic_agent": true
}
```

---

### 3. `video_ingestion_tool` — Ingest Videos

Downloads and ingests a video into the MMCT pipeline (transcription, chaptering, frame extraction, indexing).

| Parameter                  | Type           | Required          | Description                                 |
| -------------------------- | -------------- | ----------------- | ------------------------------------------- |
| `video_url`                | string         | ✅                | Video download URL                          |
| `file_name`                | string         | ✅                | Local filename for temp storage             |
| `language`                 | Languages enum | ✅                | Video language (e.g., `en-IN`, `hi-IN`)     |
| `url`                      | string         | ❌                | Source URL for metadata                     |
| `transcript_url`           | string         | ❌                | URL to download existing transcript         |
| `transcript_file_name`     | string         | ❌                | Filename for transcript                     |
| `hash_video_id`            | string         | ❌                | Custom video ID (auto-generated if omitted) |
| `frame_stacking_grid_size` | int            | ❌ (default: `4`) | Grid size for frame stacking                |

**Example:**

```json
{
  "video_url": "https://example.com/video.mp4",
  "file_name": "lecture.mp4",
  "language": "en-IN",
  "url": "www.youtube.com/watch?v=abc123"
}
```

---

### 4. `get_context_tool` — Retrieve Transcript Chunks

Searches the chapter vector index for relevant transcript segments and summaries.

| Parameter            | Type         | Required          | Description                           |
| -------------------- | ------------ | ----------------- | ------------------------------------- |
| `query`              | string       | ✅                | Search query                          |
| `video_id`           | string       | ❌                | Filter by video ID                    |
| `url`                | string       | ❌                | Filter by video URL                   |
| `fields_to_retrieve` | list[string] | ❌                | Fields to return (see defaults below) |
| `start_time`         | float        | ❌                | Filter by start time (seconds)        |
| `end_time`           | float        | ❌                | Filter by end time (seconds)          |
| `top`                | int          | ❌ (default: `3`) | Number of results                     |

**Default fields:** `chapter_transcript`, `detailed_summary`, `action_taken`, `text_from_scene`, `start_time`, `end_time`, `hash_video_id`, `url`

**Example:**

```json
{
  "query": "How to compute GCD?",
  "url": "www.youtube.com/watch?v=2W3BKOSg958",
  "top": 3
}
```

---

### 5. `get_relevant_frames_tool` — Find Frames by Visual Query

Searches keyframe embeddings to find visually relevant frames for a query. Returns frame filenames and timestamps.

| Parameter  | Type   | Required           | Description                      |
| ---------- | ------ | ------------------ | -------------------------------- |
| `query`    | string | ✅                 | Visual description to search for |
| `video_id` | string | ✅                 | Video ID to search within        |
| `top_k`    | int    | ❌ (default: `10`) | Number of frames to return       |

**Example:**

```json
{
  "query": "person writing on whiteboard",
  "video_id": "2W3BKOSg958",
  "top_k": 5
}
```

---

### 6. `query_frame_tool` — Analyze Specific Video Frames

Uses vision models to analyze video frames and answer questions. Works with either specific frame IDs or time ranges.

| Parameter    | Type   | Required | Description                                      |
| ------------ | ------ | -------- | ------------------------------------------------ |
| `query`      | string | ✅       | What to look for in frames                       |
| `frame_ids`  | list   | ❌       | Specific frame filenames to analyze              |
| `video_id`   | string | ❌       | Video ID (required with frame_ids or time range) |
| `start_time` | float  | ❌       | Start time in seconds                            |
| `end_time`   | float  | ❌       | End time in seconds                              |

> **⚠️ Important:** You must provide either `frame_ids` + `video_id` **or** `start_time` + `end_time` + `video_id`. Calling with only `query` will result in a `TypeError`.

**Example (time range):**

```json
{
  "query": "What is shown on the slide?",
  "video_id": "2W3BKOSg958",
  "start_time": 60.0,
  "end_time": 120.0
}
```

**Example (frame IDs):**

```json
{
  "query": "Describe the content",
  "frame_ids": ["2W3BKOSg958_528.jpg", "2W3BKOSg958_11214.jpg"],
  "video_id": "2W3BKOSg958"
}
```

---

### 7. `get_object_collection_tool` — Object Lookup in Videos

Retrieves details of specific objects detected in a video from the object registry index.

| Parameter      | Type         | Required | Description                |
| -------------- | ------------ | -------- | -------------------------- |
| `object_names` | list[string] | ✅       | Object names to search for |
| `video_id`     | string       | ❌       | Filter by video ID         |
| `url`          | string       | ❌       | Filter by video URL        |

> **💡 Tip:** Provide an exhaustive list of possible object names. The tool uses fuzzy matching (>0.6 similarity threshold).

**Example:**

```json
{
  "object_names": ["Presentation Slide", "whiteboard", "equations"],
  "video_id": "2W3BKOSg958"
}
```

---

### 8. `get_video_summary_tool` — Video Summary & Discovery

Retrieves high-level video summaries. Use without `video_id`/`url` for video discovery, or with them for a specific video's summary.

| Parameter  | Type   | Required          | Description                      |
| ---------- | ------ | ----------------- | -------------------------------- |
| `query`    | string | ✅                | Search query for video summaries |
| `video_id` | string | ❌                | Specific video ID                |
| `url`      | string | ❌                | Specific video URL               |
| `top`      | int    | ❌ (default: `3`) | Number of results (max 3)        |

**Example (discovery — no video_id):**

```json
{
  "query": "lectures about Python programming"
}
```

**Example (specific video):**

```json
{
  "query": "summary of this video",
  "video_id": "2W3BKOSg958",
  "top": 1
}
```

---

## Recommended Tool Usage Patterns

### Pattern 1: Full Video QA (Simplest)

Use `video_agent_tool` — it handles everything internally.

### Pattern 2: Granular Analysis Pipeline

For more control, chain the lower-level tools:

```
get_video_summary_tool   →  Discover relevant videos, get video_ids
        ↓
get_context_tool         →  Retrieve transcript chunks and timestamps
        ↓
get_relevant_frames_tool →  Find visually relevant frames
        ↓
query_frame_tool         →  Analyze specific frames with vision models
```

### Pattern 3: Object-Specific Queries

```
get_video_summary_tool       →  Get video_id
        ↓
get_object_collection_tool   →  Find detected objects and their details
        ↓
query_frame_tool             →  Visually verify objects in specific frames
```

---

## Project Structure

```
mcp_server/
├── main.py                           # Entry point — imports all tools, starts server
├── server.py                         # FastMCP instance + health check endpoint
├── config.py                         # Provider config singletons (loaded from .env)
├── client.py                         # Test client for validating tools
└── tools/
    ├── video_agent_tool.py           # End-to-end video QA
    ├── image_agent_tool.py           # Image analysis & QA
    ├── video_ingestion_tool.py       # Video ingestion pipeline
    ├── get_context_tool.py           # Transcript/chapter search
    ├── get_relevant_frames_tool.py   # Visual frame search
    ├── query_frame_tool.py           # Frame-level vision analysis
    ├── get_object_collection_tool.py # Object registry lookup
    └── get_video_summary_tool.py     # Video summary retrieval
```

---

## Connecting from an MCP Client

### Python (FastMCP Client)

```python
from fastmcp import Client
import asyncio

async def main():
    client = Client("http://<server-host>:8000/mcp")
    async with client:
        await client.ping()

        # List available tools
        tools = await client.list_tools()
        for t in tools:
            print(f"Tool: {t.name}")

        # Call a tool
        result = await client.call_tool(
            name="video_agent_tool",
            arguments={"query": "What is this video about?"}
        )
        print(result)

asyncio.run(main())
```

### Claude Desktop / Cursor / VS Code

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "mmct-agent": {
      "url": "http://<server-host>:8000/mcp"
    }
  }
}
```

---

## Troubleshooting

| Issue                                       | Solution                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `Address already in use`                    | Kill existing process on port 8000: `lsof -ti:8000 \| xargs kill`          |
| `NameError: name 'Dict' is not defined`     | Ensure `from typing import Dict, Any` is imported                          |
| `'VIT'` KeyError in image_agent             | Use **lowercase** tool names: `vit`, `ocr`, `recog`, `object_detection`    |
| `'NoneType' is not iterable` in query_frame | Must provide either `frame_ids` or `start_time`+`end_time` with `video_id` |
| Azure credential errors                     | Run `az login` or ensure Managed Identity is configured                    |
| Missing `.env` variables                    | Check all required variables are set (see Configuration section)           |
