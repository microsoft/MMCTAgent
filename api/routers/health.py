"""Health check router."""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Returns service status and current UTC timestamp.",
    response_description="Service health status",
)
async def health_check():
    """Lightweight liveness probe — no provider calls are made."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
