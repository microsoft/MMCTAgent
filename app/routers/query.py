from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from app.schemas.query import ImageQueryRequest, VideoQueryRequest
from app.services.query_services import process_image_query, process_video_query
from app.schemas.query import UnifiedQueryRequest
from app.services.query_services import process_query_v2_endpoint, process_query_v2_stream

router = APIRouter()

@router.post(
    "/query-on-images",
    summary="Query image with specified tools",
    description="Upload an image and specify which tools to run. Tools: object_detection, ocr, recog, vit.",
)
async def query_images(file: UploadFile = File(...), data: ImageQueryRequest = Depends()):
    return await process_image_query(file, data.model_dump())

@router.post("/query-on-videos")
async def query_videos(data: VideoQueryRequest = Depends()):
    return await process_video_query(data.model_dump())

@router.post(
    "/v2/query",
    summary="Unified V2 Query (Video + Image)",
    description="Unified endpoint for video and image queries using the V2 multi-agent system.",
)
async def query_unified(
    file: UploadFile = File(None),
    data: UnifiedQueryRequest = Depends()
):
    body = data.model_dump()
    
    if file:
        import tempfile
        import os
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            body["image_path"] = tmp.name
            
    return await process_query_v2_endpoint(body)


@router.post(
    "/v2/query/stream",
    summary="Streaming V2 Query (Video + Image)",
    description="Streaming endpoint that sends agent logs as Server-Sent Events (SSE).",
)
async def query_unified_stream(
    file: UploadFile = File(None),
    data: UnifiedQueryRequest = Depends()
):
    body = data.model_dump()
    
    if file:
        import tempfile
        import os
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            body["image_path"] = tmp.name
    
    return StreamingResponse(
        process_query_v2_stream(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )