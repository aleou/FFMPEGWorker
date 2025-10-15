"""Utility helpers for uploading artifacts to S3-compatible storage."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config
from loguru import logger

from app.core.config import Settings


class S3Uploader:
    """Upload processed artifacts to S3-compatible storage and issue presigned URLs."""

    def __init__(self, settings: Settings) -> None:
        if not settings.S3_BUCKET:
            raise ValueError("S3 bucket is not configured.")

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
        self._bucket = settings.S3_BUCKET
        self._presign_ttl = settings.S3_PRESIGN_TTL_SECONDS

    def upload_file(self, local_path: Path, key: str) -> None:
        content_type, _ = mimetypes.guess_type(str(local_path))
        extra_args = {"ContentType": content_type} if content_type else None
        logger.info("Uploading %s to s3://%s/%s", local_path, self._bucket, key)
        self._client.upload_file(str(local_path), self._bucket, key, ExtraArgs=extra_args or {})

    def generate_presigned_url(self, key: str, expires_in: Optional[int] = None) -> str:
        ttl = expires_in or self._presign_ttl
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=ttl,
        )

    def store_file_and_get_url(
        self,
        local_path: Path,
        key_prefix: str,
        filename: Optional[str] = None,
    ) -> str:
        key_name = filename or local_path.name
        key = f"{key_prefix.rstrip('/')}/{key_name}"
        self.upload_file(local_path, key)
        return self.generate_presigned_url(key)

