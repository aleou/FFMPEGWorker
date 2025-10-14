"""Health and readiness routes."""

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["health"])


@router.get("/health", summary="Health probe")
async def health_check() -> dict[str, str]:
    """Return a basic heartbeat payload."""

    return {"status": "ok", "service": settings.APP_NAME}


@router.get("/readiness", summary="Readiness probe")
async def readiness_check() -> dict[str, str]:
    """Expose service readiness for orchestration systems."""

    return {"status": "ready"}

