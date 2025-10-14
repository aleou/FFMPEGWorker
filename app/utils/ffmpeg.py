"""FFmpeg helper utilities."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Awaitable, Callable, Sequence

from app.core.config import settings
from app.schemas.job import JobRead

logger = logging.getLogger(__name__)

TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


def build_ffmpeg_command(job: JobRead) -> list[str]:
    """Construct an FFmpeg invocation for the supplied job."""

    params: list[str] = job.metadata.get("ffmpeg_args", []) if job.metadata else []
    command = [
        settings.FFMPEG_BINARY,
        "-y",  # overwrite outputs
        "-i",
        str(job.source_uri),
        *params,
        str(job.target_uri),
    ]
    logger.debug("Built FFmpeg command %s", command)
    return command


async def run_ffmpeg_command(
    command: Sequence[str],
    progress_callback: Callable[[float], Awaitable[None]] | None = None,
    duration_hint: float | None = None,
) -> None:
    """Execute FFmpeg asynchronously and publish progress if possible."""

    logger.info("Running FFmpeg command: %s", " ".join(command))
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    assert process.stderr
    async for raw_line in _iter_stream(process.stderr):
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        logger.debug("FFmpeg output: %s", line)
        if not progress_callback:
            continue
        progress = _extract_progress(line, duration_hint)
        if progress is not None:
            await progress_callback(progress)

    returncode = await process.wait()
    if returncode != 0:
        stdout = await process.stdout.read() if process.stdout else b""
        stderr = await process.stderr.read() if process.stderr else b""
        logger.error("FFmpeg failed with exit code %s", returncode)
        raise RuntimeError(f"ffmpeg exited with code {returncode}: {stderr.decode('utf-8', errors='ignore')}")


async def _iter_stream(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    while True:  # pragma: no cover - simple generator
        chunk = await stream.readline()
        if not chunk:
            break
        yield chunk


def _extract_progress(line: str, duration_hint: float | None) -> float | None:
    """Attempt to compute progress from FFmpeg stderr output."""

    match = TIME_RE.search(line)
    if not match or not duration_hint or duration_hint <= 0:
        return None

    hours, minutes, seconds = match.groups()
    elapsed = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    progress = min(elapsed / duration_hint, 1.0)
    return progress
