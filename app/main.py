"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.events import register_shutdown_event, register_startup_event


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    configure_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.DESCRIPTION,
        docs_url=f"{settings.API_PREFIX}{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}{settings.API_V1_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}{settings.API_V1_PREFIX}/openapi.json",
    )

    app.include_router(api_router, prefix=f"{settings.API_PREFIX}{settings.API_V1_PREFIX}")

    register_startup_event(app)
    register_shutdown_event(app)

    return app


app = create_application()

