from fastapi import APIRouter, Depends, UploadFile, File
from schemas.query import ImageQueryRequest, VideoQueryRequest
from services.query_services import process_image_query, process_video_query

router = APIRouter()

@router.post(
    "/query-on-images",
    summary="Query image with specified tools",
    description="Upload an image and specify which tools to run. Tools: object_detection, ocr, recog, vit.",
)
async def query_images(file: UploadFile = File(...), data: ImageQueryRequest = Depends()):
    return {"result": await process_image_query(file, data.model_dump())}

@router.post("/query-on-videos")
async def query_videos(data: VideoQueryRequest = Depends()):
    return {"result": await process_video_query(data.model_dump())}
