"""Test script for video upscaling using Real-ESRGAN."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.video_upscaler_service import VideoUpscalerService, UpscaleModel


def test_upscale():
    """Test video upscaling."""
    
    # Initialize upscaler
    upscaler = VideoUpscalerService(
        device="cuda",  # or "cpu" for CPU processing
        model_name=UpscaleModel.REALESR_GENERAL_X4V3,
        tile_size=0,  # Auto-detect
        half_precision=True,  # Use FP16 for speed
    )
    
    # Test with a sample video
    input_video = Path("uploads/sample_video.mp4")
    output_video = Path("outputs/upscaled/sample_video_4x.mp4")
    
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        print("Please place a test video at: uploads/sample_video.mp4")
        return
    
    print("🚀 Starting video upscaling...")
    print(f"   Input: {input_video}")
    print(f"   Output: {output_video}")
    print(f"   Model: {upscaler.model_name}")
    print(f"   Device: {upscaler.device}")
    print(f"   Tile size: {upscaler.tile_size}")
    print(f"   FP16: {upscaler.half_precision}")
    print()
    
    try:
        # Upscale video (GPU optimized)
        result = upscaler.upscale_video_gpu_optimized(
            input_path=input_video,
            output_path=output_video,
            batch_size=4,
            audio=True,
        )
        
        print(f"✅ Upscaling complete!")
        print(f"   Result: {result}")
        
        # Get file sizes
        input_size = input_video.stat().st_size / (1024 * 1024)
        output_size = result.stat().st_size / (1024 * 1024)
        
        print(f"   Input size: {input_size:.2f} MB")
        print(f"   Output size: {output_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Upscaling failed: {e}")
        import traceback
        traceback.print_exc()


def test_anime_upscale():
    """Test anime-optimized upscaling."""
    
    upscaler = VideoUpscalerService(
        device="cuda",
        model_name=UpscaleModel.REALESR_ANIME_X4,
        tile_size=0,
        half_precision=True,
    )
    
    input_video = Path("uploads/anime_sample.mp4")
    output_video = Path("outputs/upscaled/anime_sample_4x.mp4")
    
    if not input_video.exists():
        print(f"❌ Input video not found: {input_video}")
        return
    
    print("🎌 Starting anime video upscaling...")
    
    try:
        result = upscaler.upscale_video_gpu_optimized(
            input_path=input_video,
            output_path=output_video,
            batch_size=4,
            audio=True,
        )
        
        print(f"✅ Anime upscaling complete: {result}")
        
    except Exception as e:
        print(f"❌ Upscaling failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video upscaling")
    parser.add_argument(
        "--anime", 
        action="store_true", 
        help="Use anime-optimized model"
    )
    
    args = parser.parse_args()
    
    if args.anime:
        test_anime_upscale()
    else:
        test_upscale()
