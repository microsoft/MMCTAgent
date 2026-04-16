# MMCTAgent API

FastAPI web application exposing MMCTAgent pipelines as REST endpoints.

## Run

From the repository root:

```bash
uvicorn api.main:app --reload
```

Interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

| Route | Description |
|-------|-------------|
| `GET /health` | Liveness probe |
| `/video-query/*` | Query ingested videos (one-shot or SSE stream) |
| `/ingestion/*` | Ingest a new MP4 into the knowledge graph |
| `/image-query/*` | Analyze an image with configurable vision tools |

## Configuration

Provider credentials are loaded from environment variables — see `.env.example` at the repo root. Uploaded files are stored temporarily under `uploads/` and cleaned up after each request.

---

See the root [README.md](../README.md) for the full MMCTAgent framework overview.
