"""Routes for managing processing jobs."""

import json
import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.dependencies import JobServiceDep, SettingsDep, WatermarkRemovalServiceDep
from app.schemas.job import JobCreate, JobRead, JobUpdate, WatermarkRemovalConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new processing job",
    response_description="Representation of the created job.",
)
async def enqueue_job(payload: JobCreate, job_service: JobServiceDep) -> JobRead:
    """Accept a job payload and register it for downstream processing."""

    job = job_service.create_job(payload)
    return job


@router.post(
    "/upload",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a job with an uploaded file",
    response_description="Representation of the created job.",
)
async def enqueue_job_with_upload(
    job_service: JobServiceDep,
    settings: SettingsDep,
    file: UploadFile = File(..., description="Source media file to process."),
    job_type: str = Form(default="watermark_removal", description="Type of job to queue."),
    priority: int = Form(default=5, ge=1, le=10, description="Priority applied to the enqueued job."),
    target_filename: str | None = Form(
        default=None,
        description="Optional override for the output filename (defaults to '<input>_processed').",
    ),
    metadata_json: str | None = Form(
        default=None,
        description="Optional JSON-encoded metadata to attach to the job.",
    ),
    watermark_config_json: str | None = Form(
        default=None,
        description="Optional JSON-encoded watermark removal configuration.",
    ),
) -> JobRead:
    """Accept a multipart upload, persist it locally, and enqueue the corresponding job."""

    upload_token = uuid4().hex
    uploads_dir = settings.WORK_DIR / "uploads" / upload_token
    uploads_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or f"upload-{upload_token}.mp4")
    input_path = uploads_dir / original_name.name

    # Stream upload to disk to avoid loading large files into memory.
    with input_path.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await file.close()

    logger.info("Stored upload %s at %s", file.filename, input_path)
    # Determine output filename and path.
    if target_filename:
        output_name = Path(target_filename)
        if not output_name.suffix:
            output_name = output_name.with_suffix(original_name.suffix or ".mp4")
    else:
        suffix = original_name.suffix or ".mp4"
        output_name = Path(f"{original_name.stem}_processed{suffix}")

    outputs_dir = settings.WORK_DIR / "outputs" / upload_token
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / output_name.name

    # Parse optional metadata payloads.
    metadata = None
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata JSON: {exc}",
            ) from exc

    watermark_config = None
    if watermark_config_json:
        try:
            watermark_config = WatermarkRemovalConfig.model_validate_json(watermark_config_json)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid watermark configuration: {exc.errors()}",
            ) from exc

    job_payload = JobCreate(
        source_uri=input_path,
        target_uri=output_path,
        priority=priority,
        metadata=metadata,
        job_type=job_type,
        watermark_removal_config=watermark_config,
    )

    job = job_service.create_job(job_payload)
    logger.info("Enqueued job %s of type %s", job.id, job.job_type)
    return job


@router.get(
    "",
    response_model=list[JobRead],
    summary="List queued and processed jobs",
    response_description="Collection of known jobs sorted by creation time.",
)
async def list_jobs(job_service: JobServiceDep) -> list[JobRead]:
    """Return all jobs currently tracked by the service."""

    return job_service.list_jobs()


@router.get(
    "/{job_id}",
    response_model=JobRead,
    summary="Retrieve a specific job",
    response_description="Job metadata with current state.",
)
async def get_job(job_id: UUID, job_service: JobServiceDep) -> JobRead:
    """Fetch a job by identifier."""

    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


@router.patch(
    "/{job_id}",
    response_model=JobRead,
    summary="Update a job's status",
    response_description="Updated job record.",
)
async def update_job(job_id: UUID, payload: JobUpdate, job_service: JobServiceDep) -> JobRead:
    """Allow workers to report progress back to the control plane."""

    job = job_service.update_job(job_id, payload)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


@router.post(
    "/watermark-removal",
    response_model=JobRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a watermark removal job",
    response_description="Representation of the created watermark removal job.",
)
async def enqueue_watermark_removal_job(
    payload: JobCreate,
    job_service: JobServiceDep,
    watermark_service: WatermarkRemovalServiceDep
) -> JobRead:
    """Accept a watermark removal job payload and register it for processing."""

    # Validate that this is a watermark removal job
    if payload.job_type != "watermark_removal":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job type must be 'watermark_removal' for this endpoint."
        )

    if not payload.watermark_removal_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Watermark removal configuration is required."
        )

    job = job_service.create_job(payload)
    return job

