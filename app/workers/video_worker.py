"""Video processing worker leveraging FFmpeg and AI tooling."""

import logging
from typing import Awaitable, Callable

from app.schemas.job import JobRead, JobStatus, JobUpdate
from app.services.job_service import JobService
from app.utils.ffmpeg import build_ffmpeg_command, run_ffmpeg_command

logger = logging.getLogger(__name__)


class VideoProcessingWorker:
    """High-level orchestrator for video processing jobs."""

    def __init__(self, job_service: JobService) -> None:
        self._job_service = job_service

    async def process_job(self, job: JobRead) -> None:
        """Execute the job lifecycle."""

        logger.info("Processing job %s.", job.id)
        self._job_service.update_job(
            job.id,
            JobUpdate(status=JobStatus.processing, progress=0.0),
        )

        command = build_ffmpeg_command(job)
        duration_hint = None
        if job.metadata:
            duration_value = job.metadata.get("duration_seconds")
            if isinstance(duration_value, (int, float)):
                duration_hint = float(duration_value)

        await run_ffmpeg_command(
            command,
            progress_callback=self._on_progress(job),
            duration_hint=duration_hint,
        )

        self._job_service.update_job(
            job.id,
            JobUpdate(status=JobStatus.completed, progress=1.0),
        )
        logger.info("Completed job %s.", job.id)

    def _on_progress(self, job: JobRead) -> Callable[[float], Awaitable[None]]:
        async def _callback(value: float) -> None:
            self._job_service.update_job(
                job.id,
                JobUpdate(progress=value),
            )

        return _callback
