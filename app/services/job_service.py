"""Domain service orchestrating video processing jobs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional
from uuid import UUID, uuid4

from app.core.config import Settings
from app.schemas.job import JobCreate, JobRead, JobStatus, JobUpdate

logger = logging.getLogger(__name__)


class JobService:
    """Service responsible for managing job lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store: dict[UUID, JobRead] = {}
        self._storage_path = settings.WORK_DIR / "data" / "jobs_store.json"
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._refresh_store()

    def create_job(self, payload: JobCreate) -> JobRead:
        """Register a new job for processing."""

        self._refresh_store()
        job_id = uuid4()
        job = JobRead(
            id=job_id,
            source_uri=payload.source_uri,
            target_uri=payload.target_uri,
            status=JobStatus.queued,
            job_type=payload.job_type,
            metadata=payload.metadata,
            watermark_removal_config=payload.watermark_removal_config,
            result_url=None,
        )
        self._store[job_id] = job
        self._persist_store()
        return job

    def list_jobs(self) -> list[JobRead]:
        """List the currently known jobs."""

        self._refresh_store()
        return list(self._store.values())

    def get_job(self, job_id: UUID) -> Optional[JobRead]:
        """Retrieve a job by identifier."""

        self._refresh_store()
        return self._store.get(job_id)

    def update_job(self, job_id: UUID, payload: JobUpdate) -> Optional[JobRead]:
        """Update the tracked job state."""

        self._refresh_store()
        job = self._store.get(job_id)
        if not job:
            return None

        updated = job.model_copy(
            update={
                "status": payload.status or job.status,
                "progress": payload.progress if payload.progress is not None else job.progress,
                "error": payload.error or job.error,
                "result_url": payload.result_url or job.result_url,
            }
        )
        self._store[job_id] = updated
        self._persist_store()
        return updated

    def iter_pending_jobs(self) -> Iterable[JobRead]:
        """Iterate over jobs that are still in-flight."""

        self._refresh_store()
        return [job for job in self._store.values() if job.status in {JobStatus.queued, JobStatus.processing}]

    def _refresh_store(self) -> None:
        """Reload in-memory store from disk."""

        with self._lock:
            if not self._storage_path.exists():
                return
            try:
                raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.error("Failed to load job store: %s", exc)
                return

            reconstructed: dict[UUID, JobRead] = {}
            for key, value in raw.items():
                try:
                    job_id = UUID(key)
                    job = JobRead.model_validate(value)
                except Exception as exc:  # noqa: BLE001 - defensive
                    logger.warning("Skipping invalid job entry %s: %s", key, exc)
                    continue
                reconstructed[job_id] = job
            self._store = reconstructed

    def _persist_store(self) -> None:
        """Persist the current store to disk."""

        with self._lock:
            payload = {str(job_id): job.model_dump(mode="json") for job_id, job in self._store.items()}
            self._storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
