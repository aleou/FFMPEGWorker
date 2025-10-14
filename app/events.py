"""Application lifecycle hooks."""

import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)


def register_startup_event(app: FastAPI) -> None:
    """Register startup handlers."""

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Starting FFMPEG worker service.")


def register_shutdown_event(app: FastAPI) -> None:
    """Register shutdown handlers."""

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Shutting down FFMPEG worker service.")
