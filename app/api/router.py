"""Central FastAPI router wiring all API endpoints."""

from fastapi import APIRouter

from app.api.routes import health, jobs


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(jobs.router)

