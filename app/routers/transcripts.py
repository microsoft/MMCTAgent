"""Router for transcript lookup — returns full transcript content."""

from fastapi import APIRouter, Query
from app.schemas.transcripts import TranscriptLookupResponse
from app.services.transcript_lookup_service import lookup_transcript_content

router = APIRouter()


@router.get(
    "/transcripts/lookup",
    response_model=TranscriptLookupResponse,
    summary="Look up and download transcript",
    description=(
        "Given a video_id, downloads and returns the full SRT transcript "
        "content from blob storage."
    ),
    responses={
        200: {
            "description": "Transcript found",
            "content": {
                "application/json": {
                    "example": {
                        "video_id": "Dk1toyI7AJs",
                        "transcript": "1\n00:00:00,000 --> 00:00:05,000\nHello and welcome...",
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
async def get_transcript(
    video_id: str = Query(..., description="Video identifier (e.g., YouTube ID)", examples=["Dk1toyI7AJs"]),
) -> TranscriptLookupResponse:
    content = await lookup_transcript_content(video_id)
    return TranscriptLookupResponse(
        video_id=video_id,
        transcript=content,
    )
