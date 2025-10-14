"""Pydantic models for job-related payloads."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pathlib import Path
from pydantic import AnyUrl, BaseModel, Field, model_validator


class WatermarkRemovalConfig(BaseModel):
    """Configuration for watermark removal jobs."""

    transparent: bool = Field(default=False, description="Make watermark areas transparent instead of inpainting.")
    max_bbox_percent: float = Field(default=10.0, ge=1.0, le=100.0, description="Maximum percentage of image area a bounding box can cover.")
    force_format: str | None = Field(default=None, description="Force output format (PNG, WEBP, JPG, MP4, AVI).")
    overwrite: bool = Field(default=False, description="Overwrite existing output files.")


class JobStatus(str, Enum):
    """Lifecycle state of a processing job."""

    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobBase(BaseModel):
    """Common data shared across job schemas."""

    source_uri: AnyUrl | Path = Field(..., description="Location of the source video to ingest (URL or local path).")
    target_uri: AnyUrl | Path = Field(..., description="Destination where the processed video will be saved (URL or local path).")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional job-specific metadata payload.")
    job_type: str = Field(default="video_processing", description="Type of job (video_processing, watermark_removal, etc.)")
    watermark_removal_config: WatermarkRemovalConfig | None = Field(default=None, description="Configuration for watermark removal jobs.")


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
    def validate_payload(self) -> "JobUpdate":
        if self.status is None and self.progress is None and self.error is None:
            raise ValueError("At least one field must be provided when updating a job.")
        return self

