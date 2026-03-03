import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from loguru import logger
from app.routers import query, frames, transcripts, videos
from app.version import API_VERSION, BUILD_TIMESTAMP
# from app.routers import ingestion, graph_ingestion  # Disabled: not exposed in this deployment

# Configure loguru to output to stderr (uvicorn compatible)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

app = FastAPI(
    title="MMCT Agent API",
    description=(
        "Multi-modal Critical Thinking Agent Framework for image and video analysis\n\n"
        f"**Build:** {BUILD_TIMESTAMP}"
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(query.router)
# app.include_router(ingestion.router)        # Disabled: not exposed in this deployment
# app.include_router(graph_ingestion.router)   # Disabled: not exposed in this deployment
app.include_router(frames.router)
app.include_router(transcripts.router)
app.include_router(videos.router)


def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="MMCT Agent API",
        version=API_VERSION,
        description=f"""
        # Multi-modal Critical Thinking Agent Framework
        
        **Version:** {API_VERSION} | **Built:** {BUILD_TIMESTAMP}
        
        This API provides endpoints for multi-modal AI analysis including:
        
        - **Image Analysis**: Object detection, ocr, image recognition
        - **Video Analysis**: Frame extraction, video summarization, content search
        - **Document Ingestion**: Process and index documents for search
        - **Query Processing**: Natural language queries against indexed content
        
        ## Authentication
        
        The API supports both API key and managed identity authentication depending on the configured provider.
        
        ## Rate Limits
        
        Rate limiting is applied based on the configured provider limits.
        """,
        routes=app.routes,
    )

    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {"url": "https://example.com/logo.png"}

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", tags=["root"])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "MMCT Agent API",
        "version": API_VERSION,
        "build_timestamp": BUILD_TIMESTAMP,
        "description": "Multi-modal Critical Thinking Agent Framework",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mmct-agent"}


@app.get("/providers", tags=["providers"])
async def get_supported_providers():
    """Get information about supported providers."""
    return {
        "active_providers": {
            "llm": {
                "provider": "AzureLLMProvider",
                "description": "Powers Planner, Video, Critic, and Image agents",
            },
            "text_embedding": {
                "provider": "FastEmbedBGEsmallEmbeddingProvider",
                "dimensions": 384,
                "description": "Text embeddings for semantic search (local, same as ingestion)",
            },
            "image_embedding": {
                "provider": "FastEmbedQdrantCLIPEmbeddingProvider",
                "dimensions": 512,
                "description": "Image embeddings for keyframe search (local, same as ingestion)",
            },
            "graph_query": {
                "provider": "Neo4jQueryProvider",
                "description": "HNSW vector search and graph traversal on Neo4j knowledge graph",
            },
            "storage": {
                "provider": "AzureStorageProvider",
                "description": "Keyframe and media storage in Azure Blob",
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
            "embedding": ["AzureEmbeddingProvider", "OpenAIEmbeddingProvider", "FastEmbedBGEsmallEmbeddingProvider"],
            "image_embedding": ["FastEmbedQdrantCLIPEmbeddingProvider", "ClipImageEmbeddingProvider"],
            "graph": ["Neo4jQueryProvider", "Neo4jGraphProvider", "Neo4jGraphStoreProvider", "NetworkXGraphProvider"],
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
