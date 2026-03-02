"""Router for transcript blob URL lookup."""

from fastapi import APIRouter, Query
from app.schemas.transcripts import TranscriptLookupResponse
from app.services.transcript_lookup_service import lookup_transcript_url

router = APIRouter()


@router.get(
    "/transcripts/lookup",
    response_model=TranscriptLookupResponse,
    summary="Look up transcript blob URL",
    description=(
        "Given a video_id, returns the blob URL for its SRT transcript "
        "in the 'video-transcript-lively' container, if it exists."
    ),
    responses={
        200: {
            "description": "Transcript found",
            "content": {
                "application/json": {
                    "example": {
                        "video_id": "Dk1toyI7AJs",
                        "blob_url": "https://geckostorageaccount.blob.core.windows.net/video-transcript-lively/Dk1toyI7AJs/transcript.srt",
                    }
                }
            },
        },
        404: {
            "description": "Transcript not found for the given video_id",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Transcript not found for video_id 'invalid-id' (normalized: 'invalid-id')"
                    }
                }
            },
        },
    },
)
async def get_transcript_url(
    video_id: str = Query(..., description="Video identifier (e.g., YouTube ID)", examples=["Dk1toyI7AJs"]),
) -> TranscriptLookupResponse:
    url = await lookup_transcript_url(video_id)
    return TranscriptLookupResponse(
        video_id=video_id,
        blob_url=url,
    )
