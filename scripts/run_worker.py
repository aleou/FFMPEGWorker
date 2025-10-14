"""Local development worker loop."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Ensure project root is on the import path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.services.job_registry import get_job_service_instance  # noqa: E402
from app.services.watermark_removal_service import WatermarkRemovalService  # noqa: E402
from app.workers.video_worker import VideoProcessingWorker  # noqa: E402

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    settings = get_settings()
    logger.info("Launching worker in %s environment.", settings.ENVIRONMENT)

    job_service = get_job_service_instance(settings=settings)
    watermark_service = WatermarkRemovalService(
        device=settings.AI_DEVICE,
        preferred_models=settings.WATERMARK_INPAINT_MODELS,
    )
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
