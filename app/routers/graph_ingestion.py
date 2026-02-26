from fastapi import APIRouter, Depends, UploadFile, File
from app.schemas.graph_ingestion import GraphIngestionRequest
from app.services.graph_ingestion_services import ingest_graph_direct

router = APIRouter()


@router.post("/ingest-video-graph")
async def ingest_video_graph(file: UploadFile = File(...), data: GraphIngestionRequest = Depends()):
    """
    Ingest video with temporal graph extraction.
    
    Extracts events and objects from video chapters to build a temporal
    knowledge graph. Uses local fastembed models (384-dim, CPU) for embeddings.
    """
    return await ingest_graph_direct(file, data.model_dump())
