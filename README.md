# FFMPEG Worker

FastAPI project designed as the control plane for a video processing worker capable of running FFmpeg-based montage pipelines and AI-assisted post-processing workloads.

## Features
- Modular FastAPI application ready for horizontal scaling
- Clear separation between API, services, workers, and utilities
- Pydantic-based settings with environment overrides
- Structured logging with optional JSON formatting
- Dockerized and local development workflows
- Seeded testing scaffold with `pytest`
- **AI-powered watermark removal** using Florence-2 detection and LaMA inpainting

## Watermark Removal

This project includes AI-powered watermark detection and removal capabilities, similar to tools like "Sweeta" for SORA 2 videos.

### Features
- **Florence-2 Model**: Advanced vision-language model for accurate watermark detection
- **LaMA Inpainting**: State-of-the-art inpainting model for seamless watermark removal
- **Video Support**: Process both images and videos with frame-by-frame analysis
- **Flexible Configuration**: Adjustable detection sensitivity, output formats, and processing options
- **GPU Acceleration**: Automatic CUDA detection for hardware acceleration

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
    "overwrite": false
  }
}
```

#### Configuration Options
- `transparent`: Make watermark areas transparent instead of inpainting (PNG only)
- `max_bbox_percent`: Maximum percentage of image area a bounding box can cover (1-100%)
- `force_format`: Force output format (PNG, WEBP, JPG, MP4, AVI)
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
