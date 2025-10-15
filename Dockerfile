#######################################################################
# Stage 1 - Base image with CUDA, PyTorch and system dependencies
#######################################################################
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    HF_HOME=/opt/hf-cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project sources
COPY . .

#######################################################################
# Stage 2 - Pre-download AI models (LaMA, ZITS, Florence-2)
#######################################################################
FROM base AS models

RUN mkdir -p /root/.cache/torch/hub/checkpoints

RUN python - <<'PY'
from pathlib import Path

from app.core.config import Settings
from app.services.watermark_removal_service import WatermarkRemovalService

settings = Settings(
    AI_DEVICE="cpu",
    WATERMARK_INPAINT_MODELS="lama,zits,cv2",
)

service = WatermarkRemovalService(
    device="cpu",
    preferred_models=settings.WATERMARK_INPAINT_MODELS,
)

service._load_models()
cache_paths = [
    Path("/opt/hf-cache"),
    Path.home() / ".cache/torch/hub/checkpoints",
]
for path in cache_paths:
    if path.exists():
        print(f"Cached artifacts available under {path}")
PY

#######################################################################
# Stage 3 - API runtime image
#######################################################################
FROM base AS final_api

COPY --from=models /opt/hf-cache /opt/hf-cache
COPY --from=models /root/.cache/torch /root/.cache/torch

ENV HF_HOME=/opt/hf-cache \
    PATH="/workspace/.local/bin:${PATH}"

WORKDIR /workspace

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

#######################################################################
# Stage 4 - Serverless runtime image
#######################################################################
FROM base AS final_serverless

COPY --from=models /opt/hf-cache /opt/hf-cache
COPY --from=models /root/.cache/torch /root/.cache/torch

ENV HF_HOME=/opt/hf-cache \
    PATH="/workspace/.local/bin:${PATH}"

WORKDIR /workspace

CMD ["python", "-m", "app.serverless.runpod_handler"]
