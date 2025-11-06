"""API routes for video upscaling jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import ValidationError
from urllib.parse import urlparse

import httpx
from app.dependencies import JobServiceDep, SettingsDep
from app.schemas.job import JobCreate, JobRead, VideoUpscaleConfig

router = APIRouter(prefix="/upscale", tags=["upscale"])


@router.post("/", response_model=JobRead, status_code=201)
async def create_upscale_job(
    file: Annotated[UploadFile, File(description="Video file to upscale")],
    job_service: JobServiceDep,
    settings: SettingsDep,
    model: Annotated[str, Form()] = "RealESRGAN_x4plus",
    scale: Annotated[int, Form(ge=1, le=4)] = 4,
    tile_size: Annotated[int, Form(ge=0)] = 0,
    half_precision: Annotated[bool, Form()] = True,
    preserve_audio: Annotated[bool, Form()] = True,
    target_fps: Annotated[float | None, Form()] = None,
    batch_size: Annotated[int, Form(ge=1, le=16)] = 4,
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
    job_service: JobServiceDep,
    settings: SettingsDep,
    model: str = "RealESRGAN_x4plus",
    scale: int = 4,
    tile_size: int = 0,
    half_precision: bool = True,
    preserve_audio: bool = True,
    target_fps: float | None = None,
    batch_size: int = 4,
) -> JobRead:
    """Create a video upscaling job from a remote URL."""

    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Only http/https URLs are supported.")

    upload_token = uuid4().hex
    uploads_dir = settings.WORK_DIR / "uploads" / upload_token
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_stem = Path(parsed.path or "remote_video.mp4").name or "remote_video.mp4"
    if not Path(file_stem).suffix:
        file_stem = f"{file_stem}.mp4"
    source_path = uploads_dir / file_stem

    try:
        await _download_remote_file(source_url, source_path, settings.DEFAULT_TIMEOUT_SECONDS)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Failed to download source: {exc.response.text or exc.response.reason_phrase}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Download error: {exc}") from exc

    outputs_dir = settings.WORK_DIR / "outputs" / "upscaled" / upload_token
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / f"{source_path.stem}_upscaled{source_path.suffix}"

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
        source_uri=source_path,
        target_uri=output_path,
        job_type="upscale",
        upscale_config=upscale_config,
        metadata={
            "source_url": source_url,
            "download_token": upload_token,
            "original_filename": file_stem,
        },
    )

    return job_service.create_job(job_payload)


async def _download_remote_file(url: str, destination: Path, timeout_seconds: int) -> None:
    """Stream a remote file to disk without loading everything in memory."""

    timeout = httpx.Timeout(
        connect=timeout_seconds,
        read=None,
        write=None,
        pool=None,
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as buffer:
                async for chunk in response.aiter_bytes(chunk_size=1_048_576):
                    if not chunk:
                        continue
                    buffer.write(chunk)
