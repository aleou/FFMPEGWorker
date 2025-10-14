"""Local development worker loop."""

from __future__ import annotations

import asyncio
import logging

from app.core.config import settings
from app.core.logging import configure_logging
from app.services.job_service import JobService
from app.services.watermark_removal_service import WatermarkRemovalService
from app.workers.video_worker import VideoProcessingWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    logger.info("Launching worker in %s environment.", settings.ENVIRONMENT)

    job_service = JobService(settings=settings)
    watermark_service = WatermarkRemovalService(device=settings.AI_DEVICE)
    worker = VideoProcessingWorker(job_service=job_service, watermark_service=watermark_service)

    while True:
        has_work = False
        for job in job_service.iter_pending_jobs():
            has_work = True
            await worker.process_job(job)
        if not has_work:
            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Worker stopped.")

