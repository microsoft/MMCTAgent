from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from routers import query, ingestion

app = FastAPI(
    title="MMCT Agent API",
    description="Multi-modal Critical Thinking Agent Framework for image and video analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.include_router(query.router)
app.include_router(ingestion.router)


def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="MMCT Agent API",
        version="1.0.0",
        description="""
        # Multi-modal Critical Thinking Agent Framework
        
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
        tags=[
            {
                "name": "query",
                "description": "Query operations for image, video, and document analysis"
            },
            {
                "name": "ingestion",
                "description": "Document and media ingestion operations"
            }
        ]
    )
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/", tags=["root"])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "MMCT Agent API",
        "version": "1.0.0",
        "description": "Multi-modal Critical Thinking Agent Framework",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mmct-agent"}


@app.get("/providers", tags=["providers"])
async def get_supported_providers():
    """Get information about supported providers."""
    from mmct.providers.factory import provider_factory
    
    return {
        "supported_providers": provider_factory.get_supported_providers(),
        "message": "These are the currently supported providers for each service type"
    }