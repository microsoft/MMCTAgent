"""FastAPI application entry point for MMCTAgent.

Launch with:
    cd /path/to/MMCTAgent
    uvicorn api.main:app --reload

Docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, image_query, ingestion, video_query

app = FastAPI(
    title="MMCTAgent API",
    description=(
        "REST API for the Multi-Modal Critical Thinking Agent (MMCT).\n\n"
        "## Endpoints\n"
        "- **Health** — Liveness probe\n"
        "- **Video Query** — Query ingested video content (one-shot or SSE stream)\n"
        "- **Ingestion** — Ingest a new MP4 video into the knowledge graph\n"
        "- **Image Query** — Analyse an image with configurable vision tools\n\n"
        "## Notes\n"
        "All provider credentials are loaded from environment variables (see `.env.example`). "
        "Uploaded files are stored temporarily under the `uploads/` directory and deleted "
        "after each request completes."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(video_query.router, prefix="/video-query", tags=["Video Query"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["Ingestion"])
app.include_router(image_query.router, prefix="/image-query", tags=["Image Query"])
