"""RunPod serverless handler for watermark removal jobs."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod
from loguru import logger

from app.core.config import Settings, get_settings
from app.schemas.job import WatermarkRemovalConfig
from app.services.watermark_removal_service import WatermarkRemovalService
from app.utils.s3_uploader import S3Uploader

SETTINGS = get_settings()
SERVICE = WatermarkRemovalService(
    device=SETTINGS.AI_DEVICE,
    preferred_models=SETTINGS.WATERMARK_INPAINT_MODELS,
)


def _build_settings_with_overrides(overrides: dict[str, Any] | None) -> Settings:
    if not overrides:
        return SETTINGS

    mapped = {
        "S3_ACCESS_KEY_ID": overrides.get("accessId"),
        "S3_SECRET_ACCESS_KEY": overrides.get("accessSecret"),
        "S3_BUCKET": overrides.get("bucketName"),
        "S3_ENDPOINT": overrides.get("endpointUrl"),
    }
    if "region" in overrides:
        mapped["S3_REGION"] = overrides["region"]
    if "forcePathStyle" in overrides:
        mapped["S3_FORCE_PATH_STYLE"] = bool(overrides["forcePathStyle"])

    return SETTINGS.model_copy(update={k: v for k, v in mapped.items() if v is not None})


def _download_from_url(url: str, destination: Path) -> Path:
    logger.info("Downloading source from %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    return destination


def _write_base64(data: str, destination: Path) -> Path:
    logger.info("Decoding base64 payload into %s", destination)
    payload = data[data.find(",") + 1 :] if "," in data else data
    decoded = base64.b64decode(payload)
    destination.write_bytes(decoded)
    return destination


def _resolve_source(input_payload: dict[str, Any], workdir: Path) -> Path:
    source_url = input_payload.get("source") or input_payload.get("source_url")
    source_base64 = input_payload.get("source_base64")

    if not source_url and not source_base64:
        raise ValueError("Provide either 'source' (URL) or 'source_base64'.")

    filename = input_payload.get("filename")
    if not filename and source_url:
        parsed = urlparse(str(source_url))
        filename = Path(parsed.path).name
    if not filename:
        filename = "input.bin"

    destination = workdir / filename

    if source_url:
        return _download_from_url(str(source_url), destination)
    return _write_base64(str(source_base64), destination)


def _resolve_output_path(input_payload: dict[str, Any], workdir: Path, input_path: Path) -> Path:
    output_name = input_payload.get("output_filename")
    if not output_name:
        suffix = input_path.suffix or ".mp4"
        output_name = f"{input_path.stem}_processed{suffix}"
    return workdir / output_name


def _process_job(job_input: dict[str, Any], job_id: str | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        working_dir = Path(tmpdir)
        input_path = _resolve_source(job_input, working_dir)
        output_path = _resolve_output_path(job_input, working_dir, input_path)

        config_payload = job_input.get("watermark_config") or {}
        config = WatermarkRemovalConfig.model_validate(config_payload)

        result_path = SERVICE.process_file(
            input_path=input_path,
            output_path=output_path,
            transparent=config.transparent,
            max_bbox_percent=config.max_bbox_percent,
            force_format=config.force_format,
            overwrite=config.overwrite,
        )

        overrides = job_input.get("s3Config")
        uploader_settings = _build_settings_with_overrides(overrides)
        if not uploader_settings.S3_BUCKET:
            logger.warning("S3 configuration missing; returning local path.")
            return {
                "status": "success",
                "result": {
                    "local_path": str(result_path),
                    "note": "S3 settings missing; file stored only within container.",
                },
            }

        uploader = S3Uploader(uploader_settings)
        prefix = job_input.get("s3_prefix") or f"jobs/{job_id or 'manual'}"
        presigned_url = uploader.store_file_and_get_url(Path(result_path), key_prefix=prefix)

        return {
            "status": "success",
            "result": {
                "result_url": presigned_url,
                "filename": Path(result_path).name,
            },
        }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    try:
        job_input = job.get("input") or {}
        logger.info("Received RunPod job %s", job.get("id"))
        response = _process_job(job_input, job.get("id"))
        return response
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to process job %s", job.get("id"))
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
