"""Routes for managing processing jobs."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.dependencies import JobServiceDep, WatermarkRemovalServiceDep
from app.schemas.job import JobCreate, JobRead, JobUpdate

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

