"""Domain service orchestrating video processing jobs."""

from __future__ import annotations

from typing import Iterable, Optional
from uuid import UUID, uuid4

from app.core.config import Settings
from app.schemas.job import JobCreate, JobRead, JobStatus, JobUpdate


class JobService:
    """Service responsible for managing job lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store: dict[UUID, JobRead] = {}

    def create_job(self, payload: JobCreate) -> JobRead:
        """Register a new job for processing."""

        job_id = uuid4()
        job = JobRead(
            id=job_id,
            source_uri=payload.source_uri,
            target_uri=payload.target_uri,
            status=JobStatus.queued,
            metadata=payload.metadata,
        )
        self._store[job_id] = job
        return job

    def list_jobs(self) -> list[JobRead]:
        """List the currently known jobs."""

        return list(self._store.values())

    def get_job(self, job_id: UUID) -> Optional[JobRead]:
        """Retrieve a job by identifier."""

        return self._store.get(job_id)

    def update_job(self, job_id: UUID, payload: JobUpdate) -> Optional[JobRead]:
        """Update the tracked job state."""

        job = self._store.get(job_id)
        if not job:
            return None

        updated = job.model_copy(
            update={
                "status": payload.status or job.status,
                "progress": payload.progress if payload.progress is not None else job.progress,
                "error": payload.error or job.error,
            }
        )
        self._store[job_id] = updated
        return updated

    def iter_pending_jobs(self) -> Iterable[JobRead]:
        """Iterate over jobs that are still in-flight."""

        return (job for job in self._store.values() if job.status in {JobStatus.queued, JobStatus.processing})
