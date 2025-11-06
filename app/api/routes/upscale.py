"""API routes for video upscaling jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import ValidationError

from app.dependencies import JobServiceDep, SettingsDep
from app.schemas.job import JobCreate, JobRead, VideoUpscaleConfig

router = APIRouter(prefix="/upscale", tags=["upscale"])


@router.post("/", response_model=JobRead, status_code=201)
async def create_upscale_job(
    file: Annotated[UploadFile, File(description="Video file to upscale")],
    model: Annotated[str, Form()] = "RealESRGAN_x4plus",
    scale: Annotated[int, Form(ge=1, le=4)] = 4,
    tile_size: Annotated[int, Form(ge=0)] = 0,
    half_precision: Annotated[bool, Form()] = True,
    preserve_audio: Annotated[bool, Form()] = True,
    target_fps: Annotated[float | None, Form()] = None,
    batch_size: Annotated[int, Form(ge=1, le=16)] = 4,
    job_service: JobServiceDep,
    settings: SettingsDep,
) -> JobRead:
    """Create a new video upscaling job from an uploaded file."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    upload_token = uuid4().hex
    uploads_dir = settings.WORK_DIR / "uploads" / upload_token
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / file.filename
    with file_path.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await file.close()

    outputs_dir = settings.WORK_DIR / "outputs" / "upscaled" / upload_token
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"{file_path.stem}_upscaled{file_path.suffix}"

    try:
        upscale_config = VideoUpscaleConfig(
            model=model,
            scale=scale,
            tile_size=tile_size,
            half_precision=half_precision,
            preserve_audio=preserve_audio,
            target_fps=target_fps,
            batch_size=batch_size,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    job_payload = JobCreate(
        source_uri=file_path,
        target_uri=output_path,
        job_type="upscale",
        upscale_config=upscale_config,
        metadata={
            "original_filename": file.filename,
            "content_type": file.content_type,
        },
    )

    return job_service.create_job(job_payload)


@router.post("/url", response_model=JobRead, status_code=201)
async def create_upscale_job_from_url(
    source_url: str,
    model: str = "RealESRGAN_x4plus",
    scale: int = 4,
    tile_size: int = 0,
    half_precision: bool = True,
    preserve_audio: bool = True,
    target_fps: float | None = None,
    batch_size: int = 4,
    job_service: JobServiceDep,
    settings: SettingsDep,
) -> JobRead:
    """Create a video upscaling job from a remote URL."""

    upload_token = uuid4().hex
    outputs_dir = settings.WORK_DIR / "outputs" / "upscaled" / upload_token
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "upscaled_output.mp4"

    try:
        upscale_config = VideoUpscaleConfig(
            model=model,
            scale=scale,
            tile_size=tile_size,
            half_precision=half_precision,
            preserve_audio=preserve_audio,
            target_fps=target_fps,
            batch_size=batch_size,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    job_payload = JobCreate(
        source_uri=source_url,
        target_uri=output_path,
        job_type="upscale",
        upscale_config=upscale_config,
        metadata={
            "source_url": source_url,
        },
    )

    return job_service.create_job(job_payload)
