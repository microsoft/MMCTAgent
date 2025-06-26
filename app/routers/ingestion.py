from fastapi import APIRouter, Depends, UploadFile, File
from schemas.ingestion import IngestionRequest
from services.ingestion_services import ingest_direct, ingest_queue

router = APIRouter()

@router.post("/ingest-video")
async def ingest_video(file: UploadFile = File(...), data: IngestionRequest = Depends()):
    return await ingest_direct(file, data.model_dump())

@router.post("/ingest-video-queue")
async def ingest_video_queue(file: UploadFile = File(...), data: IngestionRequest = Depends()):
    return await ingest_queue(file, data.model_dump())
