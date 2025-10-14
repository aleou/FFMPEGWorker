"""Shared job service instance accessible across the application."""

from __future__ import annotations

from typing import Optional

from app.core.config import Settings, get_settings
from app.services.job_service import JobService

_job_service: Optional[JobService] = None


def get_job_service_instance(settings: Settings | None = None) -> JobService:
    """Return a singleton JobService instance."""

    global _job_service
    if _job_service is None:
        resolved_settings = settings or get_settings()
        _job_service = JobService(settings=resolved_settings)
    return _job_service
