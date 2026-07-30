"""Health check endpoint handler for FastAPI v1 API."""

from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get System Health Status",
    description="Returns the operational status, service metadata, version, and server timestamp.",
)
async def get_health() -> HealthResponse:
    """Retrieve service health status."""
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
