"""Interfaces and base classes for worker implementations."""

from __future__ import annotations

import abc
from typing import Any

from app.schemas.job import JobRead


class BaseWorker(abc.ABC):
    """Template for worker components that process jobs from the queue."""

    @abc.abstractmethod
    async def handle(self, job: JobRead) -> None:
        """Process a job payload."""

    async def on_success(self, job: JobRead) -> None:  # pragma: no cover - hooks
        """Hook executed after a successful job execution."""

    async def on_failure(self, job: JobRead, error: Exception) -> None:  # pragma: no cover - hooks
        """Hook executed when job execution fails."""

