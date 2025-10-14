#!/usr/bin/env python3
"""Test script for watermark removal functionality."""

import asyncio
from pathlib import Path
from app.services.watermark_removal_service import WatermarkRemovalService
from app.core.config import get_settings

async def test_watermark_removal():
    """Test the watermark removal service."""
    settings = get_settings()

    # Initialize service
    service = WatermarkRemovalService(device=settings.AI_DEVICE)

    # Test with a sample image (you would need to provide an actual image)
    input_path = Path("test_input.jpg")  # Replace with actual test file
    output_path = Path("test_output.png")

    if input_path.exists():
        print(f"Processing {input_path}...")
        result = service.process_file(
            input_path=input_path,
            output_path=output_path,
            transparent=False,
            max_bbox_percent=10.0,
            force_format="PNG"
        )
        print(f"Result: {result}")
    else:
        print(f"Test input file {input_path} not found. Skipping file processing test.")

    print("Watermark removal service initialized successfully!")

if __name__ == "__main__":
    asyncio.run(test_watermark_removal())