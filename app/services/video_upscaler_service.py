"""Service for AI-powered video upscaling using Real-ESRGAN on GPU."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
except ImportError:
    logger.warning("Real-ESRGAN not installed. Run: pip install realesrgan basicsr")
    RealESRGANer = None
    RRDBNet = None
    SRVGGNetCompact = None


class UpscaleModel:
    """Available upscaling models."""
    REALESR_GENERAL_X4V3 = "RealESRGAN_x4plus"  # Best quality, slower
    REALESR_ANIME_X4 = "RealESRGAN_x4plus_anime_6B"  # Optimized for anime
    REALESRNET_X4 = "RealESRNet_x4plus"  # Faster, good quality
    REALESR_ANIME_X2 = "realesr-animevideov3"  # 2x upscaling for video


class VideoUpscalerService:
    """Service for upscaling videos using Real-ESRGAN with GPU acceleration."""

    def __init__(
        self,
        device: str = "auto",
        model_name: str = UpscaleModel.REALESR_GENERAL_X4V3,
        tile_size: int = 0,  # 0 = auto, or set 256/512 for low VRAM
        tile_pad: int = 10,
        pre_pad: int = 0,
        half_precision: bool = True,
    ):
        """Initialize the video upscaler service.

        Args:
            device: Device to run models on ('cuda', 'cpu', or 'auto')
            model_name: Model to use for upscaling
            tile_size: Tile size for processing (0=auto, 256/512 for low VRAM)
            tile_pad: Padding for tiles
            pre_pad: Pre-padding for tiles
            half_precision: Use FP16 for faster processing (requires GPU)
        """
        if RealESRGANer is None:
            raise ImportError(
                "Real-ESRGAN is not installed. Install with: pip install realesrgan basicsr"
            )

        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.half_precision = half_precision and self.device.startswith("cuda")
        
        # Auto-detect optimal tile size based on GPU VRAM
        if tile_size == 0:
            tile_size = self._auto_detect_tile_size()
        
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        
        self.upsampler: Optional[RealESRGANer] = None
        self.scale_factor = 4  # Default scale
        
        logger.info(
            f"VideoUpscaler initialized: device={self.device}, model={model_name}, "
            f"tile_size={tile_size}, fp16={self.half_precision}"
        )

    def _auto_detect_tile_size(self) -> int:
        """Auto-detect optimal tile size based on GPU VRAM."""
        if not torch.cuda.is_available():
            return 400  # CPU fallback
        
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Tile size recommendations based on VRAM
            if vram_gb >= 24:  # RTX 4090, A100, etc.
                return 512
            elif vram_gb >= 16:  # RTX 4080, A40
                return 400
            elif vram_gb >= 10:  # RTX 3080, 4070
                return 300
            elif vram_gb >= 8:  # RTX 3070, 4060
                return 256
            else:  # < 8GB
                return 200
        except Exception:
            return 256  # Safe fallback

    def _load_model(self) -> None:
        """Load the Real-ESRGAN model if not already loaded."""
        if self.upsampler is not None:
            return
        
        logger.info(f"Loading Real-ESRGAN model: {self.model_name}")
        
        # Determine model path and scale
        model_path = self._get_model_path(self.model_name)
        
        # Create the appropriate network architecture
        if "anime" in self.model_name.lower():
            if "videov3" in self.model_name.lower():
                # Anime Video model (compact architecture)
                model = SRVGGNetCompact(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_conv=16, upscale=4, act_type='prelu'
                )
                netscale = 4
                self.scale_factor = 4
            else:
                # Anime x4 model (large RRDB architecture)
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4
                )
                netscale = 4
                self.scale_factor = 4
        elif "x4plus" in self.model_name.lower():
            # General x4 model
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4
            )
            netscale = 4
            self.scale_factor = 4
        else:
            # Default to x4 RRDB
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4
            )
            netscale = 4
            self.scale_factor = 4
        
        # Initialize upsampler
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half_precision,
            device=self.device,
        )
        
        logger.info(f"Model loaded successfully. Scale factor: {self.scale_factor}x")

    def _get_model_path(self, model_name: str) -> str:
        """Get the path to the model weights (downloads if needed)."""
        model_urls = {
            UpscaleModel.REALESR_GENERAL_X4V3: 
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            UpscaleModel.REALESR_ANIME_X4:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            UpscaleModel.REALESRNET_X4:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
            UpscaleModel.REALESR_ANIME_X2:
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        }
        
        if model_name not in model_urls:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Cache directory
        cache_dir = Path.home() / ".cache" / "realesrgan"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = cache_dir / f"{model_name}.pth"
        
        # Download if not exists
        if not model_path.exists():
            logger.info(f"Downloading model {model_name}...")
            import urllib.request
            urllib.request.urlretrieve(model_urls[model_name], str(model_path))
            logger.info(f"Model downloaded to {model_path}")
        
        return str(model_path)

    def upscale_image(self, image: np.ndarray) -> np.ndarray:
        """Upscale a single image using Real-ESRGAN.

        Args:
            image: Input image (RGB numpy array)

        Returns:
            Upscaled image (RGB numpy array)
        """
        self._load_model()
        
        try:
            # Real-ESRGAN expects BGR format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Upscale
            output, _ = self.upsampler.enhance(image_bgr, outscale=self.scale_factor)
            
            # Convert back to RGB
            if len(output.shape) == 3 and output.shape[2] == 3:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            return output
            
        except Exception as e:
            logger.error(f"Failed to upscale image: {e}")
            raise

    def upscale_video(
        self,
        input_path: Path,
        output_path: Path,
        fps: Optional[float] = None,
        audio: bool = True,
    ) -> Path:
        """Upscale a video file using Real-ESRGAN.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            fps: Target FPS (None = keep original)
            audio: Preserve original audio

        Returns:
            Path to upscaled video
        """
        self._load_model()
        
        logger.info(f"Starting video upscaling: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        target_fps = fps or original_fps
        output_width = width * self.scale_factor
        output_height = height * self.scale_factor
        
        logger.info(
            f"Video info: {width}x{height} -> {output_width}x{output_height}, "
            f"{total_frames} frames @ {target_fps:.2f} fps"
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            video_tmp_path = tmpdir_path / "upscaled_no_audio.mp4"
            
            # Configure video writer with hardware encoding if available
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(video_tmp_path),
                fourcc,
                target_fps,
                (output_width, output_height)
            )
            
            # Process frames
            frame_count = 0
            logger.info("Processing frames...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Upscale
                upscaled_rgb = self.upscale_image(frame_rgb)
                
                # Convert back to BGR
                upscaled_bgr = cv2.cvtColor(upscaled_rgb, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(upscaled_bgr)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            cap.release()
            out.release()
            
            logger.info("Frame processing complete. Remuxing audio...")
            
            # Remux with audio using FFmpeg
            if audio:
                self._remux_audio(video_tmp_path, input_path, output_path)
            else:
                shutil.move(str(video_tmp_path), str(output_path))
        
        logger.info(f"Video upscaling complete: {output_path}")
        return output_path

    def _remux_audio(
        self,
        video_path: Path,
        original_path: Path,
        output_path: Path
    ) -> None:
        """Remux video with original audio track."""
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video_path),      # Upscaled video
            "-i", str(original_path),    # Original (for audio)
            "-map", "0:v:0",             # Video from first input
            "-map", "1:a?",              # Audio from second input (if exists)
            "-c:v", "libx264",           # Re-encode video for compatibility
            "-preset", "slow",
            "-crf", "18",                # High quality
            "-c:a", "copy",              # Copy audio without re-encoding
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info("Audio remux successful")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Audio remux failed: {e}. Saving video without audio.")
            shutil.move(str(video_path), str(output_path))

    def upscale_video_gpu_optimized(
        self,
        input_path: Path,
        output_path: Path,
        batch_size: int = 4,
        fps: Optional[float] = None,
        audio: bool = True,
    ) -> Path:
        """GPU-optimized batch video upscaling.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            batch_size: Number of frames to process in parallel
            fps: Target FPS (None = keep original)
            audio: Preserve original audio

        Returns:
            Path to upscaled video
        """
        self._load_model()
        
        logger.info(f"Starting GPU-optimized video upscaling: {input_path}")
        
        # Use PyAV for more efficient decoding
        try:
            import av
        except ImportError:
            logger.warning("PyAV not available, falling back to standard upscaling")
            return self.upscale_video(input_path, output_path, fps, audio)
        
        # Open video
        container = av.open(str(input_path))
        video_stream = container.streams.video[0]
        
        original_fps = float(video_stream.average_rate or video_stream.base_rate or 30.0)
        target_fps = fps or original_fps
        
        width = video_stream.width
        height = video_stream.height
        output_width = width * self.scale_factor
        output_height = height * self.scale_factor
        
        logger.info(f"Video: {width}x{height} -> {output_width}x{output_height} @ {target_fps:.2f} fps")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            video_tmp_path = tmpdir_path / "upscaled_no_audio.mp4"
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(video_tmp_path),
                fourcc,
                target_fps,
                (output_width, output_height)
            )
            
            # Process in batches
            frame_batch = []
            frame_count = 0
            
            for frame in container.decode(video_stream):
                # Convert to numpy
                frame_np = frame.to_ndarray(format='rgb24')
                frame_batch.append(frame_np)
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    upscaled_batch = self._upscale_batch(frame_batch)
                    for upscaled in upscaled_batch:
                        upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                        out.write(upscaled_bgr)
                    
                    frame_count += len(frame_batch)
                    logger.info(f"Processed {frame_count} frames")
                    frame_batch = []
            
            # Process remaining frames
            if frame_batch:
                upscaled_batch = self._upscale_batch(frame_batch)
                for upscaled in upscaled_batch:
                    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
                    out.write(upscaled_bgr)
            
            container.close()
            out.release()
            
            # Remux audio
            if audio:
                self._remux_audio(video_tmp_path, input_path, output_path)
            else:
                shutil.move(str(video_tmp_path), str(output_path))
        
        logger.info(f"GPU-optimized upscaling complete: {output_path}")
        return output_path

    def _upscale_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of frames efficiently."""
        upscaled = []
        for frame in frames:
            upscaled.append(self.upscale_image(frame))
        return upscaled
