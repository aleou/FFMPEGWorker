# FFMPEG Worker

FastAPI project designed as the control plane for a video processing worker capable of running FFmpeg-based montage pipelines and AI-assisted post-processing workloads.

## 🚀 Recent Performance Improvements

**GPU Optimization Update (Oct 2025)**:
- ⚡ **3-6x faster** video processing with optimized CUDA batch processing
- 🎯 **95-99% watermark coverage** with improved temporal consistency  
- 💾 Better GPU utilization (30% → 70-85%)
- 🔧 Automatic NVENC hardware encoding with CPU fallback

See [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) and [GPU_TUNING_GUIDE.md](GPU_TUNING_GUIDE.md) for details.

## Features
- Modular FastAPI application ready for horizontal scaling
- Clear separation between API, services, workers, and utilities
- Pydantic-based settings with environment overrides
- Structured logging with optional JSON formatting
- Dockerized and local development workflows
- Seeded testing scaffold with `pytest`
- **AI-powered watermark removal** using Florence-2/YOLO detection and LaMA inpainting

## Watermark Removal

This project includes AI-powered watermark detection and removal capabilities, optimized for GPU acceleration.

### Features
- **Dual Detection**: Florence-2 (vision-language) or YOLO (fast object detection)
- **LaMA Inpainting**: State-of-the-art inpainting for seamless removal
- **Video Support**: Optimized frame-by-frame processing with temporal consistency
- **Flexible Configuration**: Adjustable detection sensitivity, batch sizes, and quality settings
- **GPU Acceleration**: CUDA-optimized with NVENC hardware encoding (when available)

### Performance (300 frames @ 1080p)
| Mode | Time | FPS | GPU Util |
|------|------|-----|----------|
| CPU (baseline) | 71s | 4.2 | 0% |
| GPU (optimized) | 12-20s | 15-25 | 70-85% |

### API Usage

#### Submit Watermark Removal Job
```bash
POST /api/v1/jobs/watermark-removal
Content-Type: application/json

{
  "source_uri": "file:///path/to/input/video.mp4",
  "target_uri": "file:///path/to/output/video_no_watermark.mp4",
  "job_type": "watermark_removal",
  "watermark_removal_config": {
    "transparent": false,
    "max_bbox_percent": 10.0,
    "force_format": "MP4",
    "detector": "yolo",
    "overwrite": false
  }
}
```

#### Configuration Options
- `transparent`: Make watermark areas transparent instead of inpainting (PNG only)
- `max_bbox_percent`: Maximum percentage of image area a bounding box can cover (1-100%)
- `force_format`: Force output format (PNG, WEBP, JPG, MP4, AVI)
- `detector`: Detection backend - "yolo" (faster) or "florence" (more accurate)
- `overwrite`: Overwrite existing output files

### Running the Worker
```bash
python scripts/run_worker.py
```

The worker will automatically process watermark removal jobs alongside standard video processing jobs.

## Getting Started

### 1. Configure Python Environment
```bash
python -m venv venv
source venv/bin/activate        # PowerShell: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file based on the sample.
```bash
cp .env.example .env
```

### 3. Run the API
```bash
uvicorn app.main:app --reload
```

The interactive docs will be available at `http://localhost:8000/api/v1/docs`.

## Project Layout
```
app/
├── api/            # FastAPI routers
├── core/           # Settings, logging, bootstrap
├── services/       # Domain logic
├── workers/        # Background workers
├── schemas/        # Pydantic models
└── utils/          # Shared helpers
tests/              # Pytest suite
scripts/            # CLI / automation utilities
```

## Docker Workflow

### Build image
```bash
docker build -t ffmpeg-worker-api .
```

### Run container
```bash
docker run --rm -p 8000:8000 --env-file .env ffmpeg-worker-api
```

## Development Tooling
- `ruff` for linting & formatting
- `mypy` for static analysis
- `pytest` (with `pytest-asyncio`) for testing

## Next Steps
- Plug in the queue and persistence layer suited to your stack
- Implement real FFmpeg progress tracking and AI pipelines
- Harden telemetry and observability
