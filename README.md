# FFMPEG Worker

FastAPI project designed as the control plane for a video processing worker capable of running FFmpeg-based montage pipelines and AI-assisted post-processing workloads.

## Features
- Modular FastAPI application ready for horizontal scaling
- Clear separation between API, services, workers, and utilities
- Pydantic-based settings with environment overrides
- Structured logging with optional JSON formatting
- Dockerized and local development workflows
- Seeded testing scaffold with `pytest`

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
