"""Utility helpers for interacting with S3-compatible storage."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config
from loguru import logger

from app.core.config import Settings


class S3Uploader:
    """Upload/download artifacts to S3-compatible storage and issue presigned URLs."""

    def __init__(self, settings: Settings) -> None:
        if not settings.S3_BUCKET:
            logger.warning("S3 bucket not configured; uploads require explicit bucket.")

        session_kwargs: dict[str, object] = {}
        if settings.S3_ENDPOINT:
            session_kwargs["endpoint_url"] = settings.S3_ENDPOINT
        if settings.S3_REGION:
            session_kwargs["region_name"] = settings.S3_REGION
        if settings.S3_ACCESS_KEY_ID and settings.S3_SECRET_ACCESS_KEY:
            session_kwargs["aws_access_key_id"] = settings.S3_ACCESS_KEY_ID
            session_kwargs["aws_secret_access_key"] = settings.S3_SECRET_ACCESS_KEY

        addressing_style = "path" if settings.S3_FORCE_PATH_STYLE else "auto"
        session_kwargs["config"] = Config(s3={"addressing_style": addressing_style})

        self._client = boto3.client("s3", **session_kwargs)
        self._default_bucket = settings.S3_BUCKET
        self._presign_ttl = settings.S3_PRESIGN_TTL_SECONDS

    def upload_file(self, local_path: Path, key: str, bucket: Optional[str] = None) -> None:
        target_bucket = bucket or self._default_bucket
        if not target_bucket:
            raise ValueError("S3 bucket must be provided to upload files.")

        content_type, _ = mimetypes.guess_type(str(local_path))
        extra_args = {"ContentType": content_type} if content_type else None
        logger.info("Uploading %s to s3://%s/%s", local_path, target_bucket, key)
        self._client.upload_file(str(local_path), target_bucket, key, ExtraArgs=extra_args or {})

    def generate_presigned_url(
        self,
        key: str,
        expires_in: Optional[int] = None,
        bucket: Optional[str] = None,
    ) -> str:
        ttl = expires_in or self._presign_ttl
        target_bucket = bucket or self._default_bucket
        if not target_bucket:
            raise ValueError("S3 bucket must be provided to generate presigned URLs.")

        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": target_bucket, "Key": key},
            ExpiresIn=ttl,
        )

    def download_to_path(
        self,
        key: str,
        destination: Path,
        bucket: Optional[str] = None,
    ) -> Path:
        target_bucket = bucket or self._default_bucket
        if not target_bucket:
            raise ValueError("S3 bucket must be provided to download files.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s to %s", target_bucket, key, destination)
        self._client.download_file(target_bucket, key, str(destination))
        return destination

    def store_file_and_get_url(
        self,
        local_path: Path,
        key_prefix: str,
        filename: Optional[str] = None,
        bucket: Optional[str] = None,
        expires_in: Optional[int] = None,
    ) -> str:
        key_name = filename or local_path.name
        key = f"{key_prefix.rstrip('/')}/{key_name}"
        self.upload_file(local_path, key, bucket=bucket)
        return self.generate_presigned_url(key, expires_in=expires_in, bucket=bucket)
