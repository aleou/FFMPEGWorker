"""Application configuration and settings management."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application configuration."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "FFMPEG Worker"
    APP_VERSION: str = "0.1.0"
    DESCRIPTION: str = (
        "High-throughput worker that orchestrates video processing workloads "
        "and AI-powered montage pipelines."
    )
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    API_PREFIX: str = "/api"
    API_V1_PREFIX: str = "/v1"

    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False

    WORK_DIR: Path = Path.cwd()
    FFMPEG_BINARY: str = "ffmpeg"
    WORKER_CONCURRENCY: int = 2
    JOB_QUEUE_BROKER_URL: str | None = None
    JOB_QUEUE_BACKEND_URL: str | None = None

    DATABASE_URL: str = "sqlite+aiosqlite:///./ffmpeg-worker.db"
    DEFAULT_TIMEOUT_SECONDS: int = 30


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()


settings = get_settings()
