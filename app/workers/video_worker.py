"""Video processing worker leveraging FFmpeg and AI tooling."""

import logging
import os
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse, unquote

from app.schemas.job import JobRead, JobStatus, JobUpdate
from app.services.job_service import JobService
from app.services.watermark_removal_service import WatermarkRemovalService
from app.utils.ffmpeg import build_ffmpeg_command, run_ffmpeg_command
from app.utils.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)


class VideoProcessingWorker:
    """High-level orchestrator for video processing jobs."""

    def __init__(
        self,
        job_service: JobService,
        watermark_service: WatermarkRemovalService | None = None,
        storage_uploader: S3Uploader | None = None,
    ) -> None:
        self._job_service = job_service
        self._watermark_service = watermark_service
        self._storage_uploader = storage_uploader

    async def process_job(self, job: JobRead) -> None:
        """Execute the job lifecycle."""

        logger.info("Processing job %s of type %s.", job.id, job.job_type)

        self._job_service.update_job(
            job.id,
            JobUpdate(status=JobStatus.processing, progress=0.0),
        )

        try:
            if job.job_type == "watermark_removal":
                await self._process_watermark_removal_job(job)
            else:
                # Default video processing
                await self._process_video_job(job)

            self._job_service.update_job(
                job.id,
                JobUpdate(status=JobStatus.completed, progress=1.0),
            )
            logger.info("Completed job %s.", job.id)
            

        except Exception as e:
            logger.error("Failed to process job %s: %s", job.id, str(e))
            self._job_service.update_job(
                job.id,
                JobUpdate(status=JobStatus.failed, error=str(e)),
            )

    async def _process_video_job(self, job: JobRead) -> None:
        """Process a standard video processing job."""
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

    async def _process_watermark_removal_job(self, job: JobRead) -> None:
        """Process a watermark removal job."""
        if not self._watermark_service:
            raise ValueError("Watermark removal service not available")

        if not job.watermark_removal_config:
            raise ValueError("Watermark removal configuration missing")

        input_path = self._resolve_path(job.source_uri)
        output_path = self._resolve_path(job.target_uri)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config = job.watermark_removal_config

        # Process the file
        result_path = self._watermark_service.process_file(
            input_path=input_path,
            output_path=output_path,
            transparent=config.transparent,
            max_bbox_percent=config.max_bbox_percent,
            force_format=config.force_format,
            detector=config.detector,
            overwrite=config.overwrite
        )

        logger.info("Watermark removal completed: %s -> %s", input_path, result_path)

        if self._storage_uploader:
            try:
                result_url = self._storage_uploader.store_file_and_get_url(
                    Path(result_path),
                    key_prefix=f"jobs/{job.id}",
                )
                logger.info("Uploaded result for job %s to %s", job.id, result_url)
                self._job_service.update_job(job.id, JobUpdate(result_url=result_url))
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to upload result for job %s: %s", job.id, exc)

    def _on_progress(self, job: JobRead) -> Callable[[float], Awaitable[None]]:
        async def _callback(value: float) -> None:
            self._job_service.update_job(
                job.id,
                JobUpdate(progress=value),
            )

        return _callback

    @staticmethod
    def _resolve_path(value: str | Path) -> Path:
        """Normalize job URIs into local filesystem paths."""

        if isinstance(value, Path):
            return value

        parsed = urlparse(str(value))
        if parsed.scheme and parsed.path:
            path_str = unquote(parsed.path)
            if os.name == "nt" and path_str.startswith("/") and len(path_str) > 1:
                # Trim the leading slash so Windows drive letters are preserved.
                path_str = path_str.lstrip("/")
            return Path(path_str)

        return Path(str(value))
