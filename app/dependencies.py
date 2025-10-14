"""Shared dependency injections for FastAPI routes."""

from collections.abc import AsyncIterator, Iterator
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.job_service import JobService
from app.services.watermark_removal_service import WatermarkRemovalService


def get_app_settings() -> Settings:
    """Provide application settings."""

    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def get_job_service(settings: SettingsDep) -> Iterator[JobService]:
    """Provide a job service instance for request handlers."""

    service = JobService(settings=settings)
    yield service


JobServiceDep = Annotated[JobService, Depends(get_job_service)]


def get_watermark_removal_service(settings: SettingsDep) -> Iterator[WatermarkRemovalService]:
    """Provide a watermark removal service instance for request handlers."""

    service = WatermarkRemovalService(device=settings.AI_DEVICE)
    yield service


WatermarkRemovalServiceDep = Annotated[WatermarkRemovalService, Depends(get_watermark_removal_service)]

