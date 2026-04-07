"""MMCT Agent API — application entry point.

Configures the FastAPI application, registers middleware, mounts routers, and
runs startup tasks (video catalog generation).
"""

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from loguru import logger

from app.routers import frames, query, transcripts, videos
from app.utilities.request_id_middleware import RequestIDMiddleware, MMCT_REQUEST_ID_HEADER
from app.version import API_VERSION, BUILD_TIMESTAMP

# ---------------------------------------------------------------------------
# Logging — structured output compatible with uvicorn
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MMCT Agent API",
    description=(
        "Multi-modal Critical Thinking Agent — video and image question-answering "
        "powered by a temporal knowledge graph.\n\n"
        f"**Build:** {BUILD_TIMESTAMP}"
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — allow all origins (tighten per-deployment as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Request correlation ID — runs after CORS so every request gets an ID
app.add_middleware(RequestIDMiddleware)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(query.router)
app.include_router(frames.router)
app.include_router(transcripts.router)
app.include_router(videos.router)

# Ingestion endpoints are intentionally not mounted in the query-serving
# deployment.  Uncomment to enable:
# from app.routers import ingestion, graph_ingestion
# app.include_router(ingestion.router)
# app.include_router(graph_ingestion.router)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_generate_video_catalog():
    """Pre-generate the video catalog at startup (best-effort, non-fatal)."""
    try:
        from app.config import get_neo4j_query_provider, get_video_agent_provider
        from app.services.video_catalog_service import generate_video_catalog

        neo4j_provider = get_neo4j_query_provider()
        llm_provider = get_video_agent_provider().llm_provider
        catalog = await generate_video_catalog(neo4j_provider, llm_provider)
        if catalog:
            logger.info(f"Video catalog ready ({len(catalog)} chars)")
        else:
            logger.warning("Video catalog is empty — agent will run without it")
    except Exception as exc:
        logger.warning(f"Video catalog generation failed at startup (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# OpenAPI schema
# ---------------------------------------------------------------------------

def custom_openapi():
    """Return a customised OpenAPI schema with MMCT branding."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title="MMCT Agent API",
        version=API_VERSION,
        description=f"""
# Multi-modal Critical Thinking Agent

**Version:** {API_VERSION} | **Built:** {BUILD_TIMESTAMP}

Endpoints for video and image analysis backed by a temporal knowledge graph:

- **Video Q&A** — natural language queries answered from ingested video content
- **Image Analysis** — object detection, OCR, recognition, and visual reasoning
- **Frame Lookup** — retrieve uniform frames by video ID and timestamp
- **Transcript Lookup** — fetch stored SRT transcripts by video ID
- **Video Catalog** — list ingested videos and browse the content catalog

## Authentication

Authentication is handled via Azure Managed Identity or API key depending on
the deployment configuration.

## Request Correlation

Every request is assigned a unique ID returned in the `{MMCT_REQUEST_ID_HEADER}`
response header. Supply the same header on inbound requests to propagate your
own correlation IDs through the system.
        """,
        routes=app.routes,
    )

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


# ---------------------------------------------------------------------------
# Built-in routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["root"])
async def root():
    """API root — returns service identity and documentation links."""
    return {
        "service": "MMCT Agent API",
        "version": API_VERSION,
        "build_timestamp": BUILD_TIMESTAMP,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Liveness probe — returns healthy if the application is running."""
    return {"status": "healthy", "service": "mmct-agent"}


@app.get("/model", tags=["model"])
async def get_model_info():
    """Return the LLM deployment currently in use."""
    from app.config import get_settings

    settings = get_settings()
    return {
        "provider": "AzureLLMProvider",
        "model_name": settings.llm_model_name,
        "deployment_name": settings.llm_deployment_name,
        "base_url": settings.llm_endpoint,
        "api_version": settings.llm_api_version,
    }


@app.get("/providers", tags=["providers"])
async def get_supported_providers():
    """Return information about the active and supported provider configurations."""
    return {
        "active_providers": {
            "llm": {
                "provider": "AzureLLMProvider",
                "description": "Powers Planner, Video, Critic, and Image agents",
            },
            "text_embedding": {
                "provider": "FastEmbedBGEsmallEmbeddingProvider",
                "dimensions": 384,
                "description": "Local text embeddings — same model used during ingestion",
            },
            "image_embedding": {
                "provider": "FastEmbedQdrantCLIPEmbeddingProvider",
                "dimensions": 512,
                "description": "Local image embeddings — same model used during ingestion",
            },
            "graph_query": {
                "provider": "Neo4jQueryProvider",
                "description": "HNSW vector search and graph traversal on the knowledge graph",
            },
            "storage": {
                "provider": "AzureStorageProvider",
                "description": "Keyframe and transcript storage in Azure Blob",
            },
            "vector_search": {
                "providers": [
                    "AISearchChapterProvider",
                    "AISearchKeyframesProvider",
                    "AISearchObjectCollectionProvider",
                ],
                "description": "Azure AI Search indexes for chapters, keyframes, and objects",
            },
        },
        "all_supported_providers": {
            "llm": ["AzureLLMProvider", "AzureReasoningLLMProvider", "OpenAILLMProvider"],
            "embedding": [
                "AzureEmbeddingProvider",
                "OpenAIEmbeddingProvider",
                "FastEmbedBGEsmallEmbeddingProvider",
            ],
            "image_embedding": [
                "FastEmbedQdrantCLIPEmbeddingProvider",
                "ClipImageEmbeddingProvider",
            ],
            "graph": [
                "Neo4jQueryProvider",
                "Neo4jGraphProvider",
                "Neo4jGraphStoreProvider",
                "NetworkXGraphProvider",
            ],
            "storage": ["AzureStorageProvider", "LocalStorageProvider"],
            "vector_db": [
                "AISearchChapterProvider",
                "AISearchKeyframesProvider",
                "AISearchObjectCollectionProvider",
                "LocalFaissSearchProvider",
                "GraphRagSearchProvider",
            ],
            "transcription": [
                "AzureSpeechServiceProvider",
                "AzureWhisperTranscriptionProvider",
                "OpenAITranscriptionProvider",
            ],
            "vision": ["AzureVisionProvider", "OpenAIVisionProvider"],
        },
    }
