"""Pydantic models for job-related payloads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, model_validator


class JobStatus(str, Enum):
    """Lifecycle state of a processing job."""

    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobBase(BaseModel):
    """Common data shared across job schemas."""

    source_uri: HttpUrl = Field(..., description="Location of the source video to ingest.")
    target_uri: HttpUrl = Field(..., description="Destination where the processed video will be saved.")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional job-specific metadata payload.")


class JobCreate(JobBase):
    """Schema for requests that create a new job."""

    priority: int = Field(default=5, ge=1, le=10, description="Higher numbers receive more attention from workers.")


class JobRead(JobBase):
    """Schema representing the stored state of a job."""

    id: UUID
    status: JobStatus
    created_at: datetime = Field(default_factory=datetime.utcnow)
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    error: Optional[str] = None


class JobUpdate(BaseModel):
    """Schema for worker-driven job updates."""

    status: Optional[JobStatus] = None
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    error: Optional[str] = None

    @model_validator(mode="after")
    def validate_payload(cls, data: "JobUpdate") -> "JobUpdate":
        if data.status is None and data.progress is None and data.error is None:
            raise ValueError("At least one field must be provided when updating a job.")
        return data

