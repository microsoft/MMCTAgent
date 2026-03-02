"""Router for frame lookup — returns actual frame bytes."""

from fastapi import APIRouter, Query
from app.schemas.frames import FrameLookupResponse, FrameHit
from app.services.frame_lookup_service import lookup_frame_bytes

router = APIRouter()


@router.get(
    "/frames/lookup",
    response_model=FrameLookupResponse,
    summary="Look up and download a uniform frame",
    description=(
        "Given a video_id and timestamp (seconds), returns the base64-encoded "
        "JPEG frame image at that exact timestamp from blob storage."
    ),
    responses={
        200: {
            "description": "Frame image found",
            "content": {
                "application/json": {
                    "example": {
                        "video_id": "Dk1toyI7AJs",
                        "requested_timestamp": 120,
                        "frames": [
                            {"timestamp_second": 120, "image_base64": "<base64>", "content_type": "image/jpeg"},
                        ],
                    }
                }
            },
        },
    },
)
async def get_frame_bytes(
    video_id: str = Query(..., description="Video identifier (e.g., YouTube ID)", examples=["Dk1toyI7AJs"]),
    timestamp: int = Query(..., ge=0, description="Centre timestamp in seconds", examples=[120]),
) -> FrameLookupResponse:
    hits = await lookup_frame_bytes(video_id, timestamp)
    return FrameLookupResponse(
        video_id=video_id,
        requested_timestamp=timestamp,
        frames=[FrameHit(timestamp_second=ts, image_base64=data) for ts, data in hits],
    )
