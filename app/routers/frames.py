"""Router for frame blob URL lookup."""

from fastapi import APIRouter, Query
from app.schemas.frames import FrameLookupResponse, FrameHit
from app.services.frame_lookup_service import lookup_frame_urls

router = APIRouter()


@router.get(
    "/frames/lookup",
    response_model=FrameLookupResponse,
    summary="Look up uniform-frame blob URLs",
    description=(
        "Given a video_id and timestamp (seconds), returns blob URLs for "
        "frames at t-1, t, and t+1 that actually exist in the "
        "'video-frames-lively' container."
    ),
    responses={
        200: {
            "description": "Frame URLs found",
            "content": {
                "application/json": {
                    "example": {
                        "video_id": "Dk1toyI7AJs",
                        "requested_timestamp": 120,
                        "frames": [
                            {"timestamp_second": 119, "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-frames-lively/Dk1toyI7AJs/119/frame.jpg"},
                            {"timestamp_second": 120, "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-frames-lively/Dk1toyI7AJs/120/frame.jpg"},
                            {"timestamp_second": 121, "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-frames-lively/Dk1toyI7AJs/121/frame.jpg"},
                        ],
                    }
                }
            },
        },
    },
)
async def get_frame_urls(
    video_id: str = Query(..., description="Video identifier (e.g., YouTube ID)", examples=["Dk1toyI7AJs"]),
    timestamp: int = Query(..., ge=0, description="Centre timestamp in seconds", examples=[120]),
) -> FrameLookupResponse:
    hits = await lookup_frame_urls(video_id, timestamp)
    return FrameLookupResponse(
        video_id=video_id,
        requested_timestamp=timestamp,
        frames=[FrameHit(timestamp_second=ts, blob_url=url) for ts, url in hits],
    )
