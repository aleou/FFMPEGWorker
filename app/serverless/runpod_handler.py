"""RunPod serverless handler for watermark removal jobs."""

from __future__ import annotations

import base64
import hashlib
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


def _download_from_url(
    url: str,
    destination: Path,
    verify: bool = True,
    headers: dict[str, str] | None = None,
) -> Path:
    logger.info("Downloading source from %s", url)
    response = requests.get(url, stream=True, timeout=60, verify=verify, headers=headers or {})
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(decoded)
    return destination


def _safe_filename(candidate: str | None, default: str = "input.bin") -> str:
    if not candidate:
        return default

    candidate = candidate.strip().strip("/\\")
    if not candidate:
        return default

    # Extract extension if present
    suffix = Path(candidate).suffix
    stem = Path(candidate).stem or "file"

    # Replace problematic characters
    cleaned_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem)
    cleaned = f"{cleaned_stem}{suffix}"

    max_len = 120
    if len(cleaned) <= max_len:
        return cleaned

    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:12]
    truncated_stem = cleaned_stem[: max(10, max_len - len(suffix) - len(digest) - 1)]
    return f"{truncated_stem}-{digest}{suffix}"


def _resolve_source(
    input_payload: dict[str, Any],
    workdir: Path,
    storage_settings: Settings,
) -> Path:
    source_url = input_payload.get("source") or input_payload.get("source_url")
    source_base64 = input_payload.get("source_base64")
    source_s3 = input_payload.get("source_s3")
    source_s3_key = input_payload.get("source_s3_key")

    if not any([source_url, source_base64, source_s3, source_s3_key]):
        raise ValueError("Provide either 'source' (URL), 'source_base64', or S3 reference.")

    filename = input_payload.get("filename")
    if not filename and source_url:
        parsed = urlparse(str(source_url))
        filename = Path(parsed.path).name or parsed.netloc
    filename = _safe_filename(filename)

    destination = workdir / filename

    if source_s3_key or source_s3 or (source_url and urlparse(str(source_url)).scheme == "s3"):
        s3_spec = source_s3 or {}
        key = source_s3_key or s3_spec.get("key")
        if not key and source_url:
            parsed = urlparse(str(source_url))
            key = parsed.path.lstrip("/")
            if not filename and parsed.path:
                candidate = Path(parsed.path).name
                destination = workdir / _safe_filename(candidate)

        bucket = (
            s3_spec.get("bucket")
            or storage_settings.S3_BUCKET
            or (urlparse(str(source_url)).netloc if source_url else None)
        )
        if not key:
            raise ValueError("S3 key is required when using S3 source.")
        uploader = S3Uploader(storage_settings)
        return uploader.download_to_path(key=key, destination=destination, bucket=bucket)

    if source_url:
        verify = bool(input_payload.get("source_verify_ssl", True))
        headers = input_payload.get("source_headers") or None
        return _download_from_url(str(source_url), destination, verify=verify, headers=headers)
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
        overrides = job_input.get("s3Config")
        uploader_settings = _build_settings_with_overrides(overrides)

        input_path = _resolve_source(job_input, working_dir, uploader_settings)
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
