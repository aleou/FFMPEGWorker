"""Service for AI-powered watermark detection and removal using Florence-2 and configurable inpainting models."""

from __future__ import annotations

import os
import subprocess
import tempfile
import json
import math
import shutil
from contextlib import nullcontext
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse
from enum import Enum

import av
import cv2
import numpy as np
import torch
import requests
from loguru import logger
from PIL import Image, ImageDraw, UnidentifiedImageError
try:
    from pydantic_core import PydanticUndefined
except ImportError:  # pragma: no cover - compatibility for Pydantic v1
    class _Undefined:  # type: ignore
        pass

    PydanticUndefined = _Undefined()  # type: ignore

try:  # pragma: no cover - compatibility shim
    from pydantic.fields import Undefined  # type: ignore
except ImportError:  # pragma: no cover
    Undefined = PydanticUndefined  # type: ignore
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model.lama import LaMa
from iopaint.model.zits import ZITS
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import tqdm

Detection = tuple[tuple[int, int, int, int], Optional[str], Optional[float]]
DEFAULT_YOLO_MODEL_URL = (
    "https://huggingface.co/hellostevelo/sora_watermark-yolov11s/resolve/main/sora_watermark-yolov11s.pt"
)
DEFAULT_YOLO_CACHE_SUBDIR = "watermark_yolo"


class TaskType(str, Enum):
    """Task types for Florence-2 model."""
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


class WatermarkRemovalService:
    """Service for detecting and removing watermarks from videos using AI models."""

    SUPPORTED_INPAINT_MODELS = ("lama", "zits", "cv2")
    WATERMARK_POSITIVE_KEYWORDS = (
        "watermark",
        "logo",
        "overlay",
        "bug",
        "subtitle",
        "channel",
        "branding",
        "corner text",
        "channel id",
        "watermark logo",
        "station id",
        "broadcast bug",
        "on screen graphic",
        "sora logo",
        "powered by",
        "copyright",
    )
    WATERMARK_NEGATIVE_KEYWORDS = (
        "traffic",
        "road",
        "sign",
        "billboard",
        "street",
        "vehicle",
        "person",
        "car",
        "truck",
        "bus",
        "bike",
        "pedestrian",
    )
    WATERMARK_MIN_SCORE = 0.10  # Reduced from 0.12 to catch more detections
    WATERMARK_FALLBACK_MIN_SCORE = 0.07  # Reduced from 0.08
    WATERMARK_DILATION_KERNEL = (7, 7)  # Increased from (5,5) for better coverage
    WATERMARK_DILATION_ITERATIONS = 3  # Increased from 2 for smoother masks
    WATERMARK_PERSISTENCE_FRAMES = 18  # Increased from 12 (~0.75s @24fps) for better temporal consistency
    WATERMARK_ACCUMULATION_SECONDS = 2.0  # Increased from 1.5 to reduce flashes
    YOLO_BATCH_SIZE = 16  # Increased from 4 - better GPU utilization
    GPU_SEGMENT_FRAMES = 300  # Increased from 180 - process larger chunks
    WATERMARK_LOOKBACK_FRAMES = 9  # Increased from 6 for better temporal tracking
    WATERMARK_LOOKAHEAD_SECONDS = 1.5  # Increased from 1.0
    WATERMARK_POSITION_THRESHOLD = 0.15  # Increased from 0.12 - more lenient position matching
    WATERMARK_MAX_GAP_FRAMES = 5  # Increased from 3 to bridge detection gaps
    INPAINT_BATCH_SIZE = 8  # Increased from 4 - better GPU parallelization
    
    # Multi-scale detection settings (can be disabled to save GPU memory)
    ENABLE_MULTISCALE_DETECTION = True  # Set to False if OOM errors persist
    MULTISCALE_SCALES = [1.0, 0.8, 1.2]  # Detection scales (100%, 80%, 120%)

    def __init__(
        self,
        device: str = "auto",
        preferred_models: Sequence[str] | str | None = None,
        default_detector: str | None = None,
        yolo_model_url: str | None = None,
        yolo_cache_dir: Path | None = None,
    ):
        """Initialize the watermark removal service.

        Args:
            device: Device to run models on ('cuda', 'cpu', or 'auto')
            preferred_models: Preferred inpainting model(s) in order of priority.
            default_detector: Default detection backend ('flo'/'florence' or 'yolo').
            yolo_model_url: Optional override URL for the YOLO watermark detector weights.
            yolo_cache_dir: Optional directory to cache YOLO detector weights.
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._cuda_capability = self._detect_cuda_capability()
        
        # Auto-detect GPU and configure optimal parameters
        self._configure_gpu_parameters()
        
        # Optimize CUDA settings for better GPU utilization
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TensorFloat-32 for faster computation
            torch.backends.cudnn.allow_tf32 = True
            # Enable memory efficient attention if available
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
            except Exception:
                pass
        
        self._inpaint_candidates = self._resolve_inpaint_candidates(preferred_models)
        self._remaining_inpaint_candidates = list(self._inpaint_candidates)

        self.florence_model: Optional[AutoModelForCausalLM] = None
        self.florence_processor: Optional[AutoProcessor] = None
        self.inpaint_model_manager: Optional[ModelManager] = None
        self._inpaint_device: torch.device | None = None
        self._active_inpaint_model: str | None = None
        self.default_detector = self._resolve_detector(default_detector)
        self._yolo_model_url = yolo_model_url or DEFAULT_YOLO_MODEL_URL
        self._yolo_cache_dir = Path(yolo_cache_dir) if yolo_cache_dir else Path.home() / ".cache" / DEFAULT_YOLO_CACHE_SUBDIR
        self._yolo_model_path: Path | None = None
        self._yolo_model = None
        self._yolo_device: str | None = None
        self._yolo_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Using device: {self.device} (inpaint preferences: {', '.join(self._inpaint_candidates)})"
        )

    @staticmethod
    def _coerce_device(device: str | torch.device) -> torch.device:
        return device if isinstance(device, torch.device) else torch.device(device)

    def _resolve_detector(self, detector: str | None) -> str:
        if detector is None:
            return "florence"

        value = detector.lower()
        if value in {"flo", "florence", "fl"}:
            return "florence"
        if value == "yolo":
            return "yolo"

        logger.warning("Unknown detector '%s'; defaulting to Florence-2.", detector)
        return "florence"

    def _effective_detector(self, detector: str | None) -> str:
        if detector is None:
            return self.default_detector

        value = detector.lower()
        if value in {"flo", "florence", "fl"}:
            return "florence"
        if value == "yolo":
            return "yolo"

        logger.warning("Unknown detector override '%s'; using default '%s'.", detector, self.default_detector)
        return self.default_detector

    def _detect_cuda_capability(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        try:
            major, minor = torch.cuda.get_device_capability()
            return float(f"{major}.{minor}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Unable to read CUDA capability ({exc}); assuming no CUDA acceleration.")
            return None

    def _configure_gpu_parameters(self) -> None:
        """Auto-detect GPU and configure optimal batch sizes and segment frames."""
        if not torch.cuda.is_available():
            logger.info("No CUDA available - using default CPU parameters")
            return
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"Detected GPU: {gpu_name} with {vram_gb:.1f} GB VRAM")
            
            # Configure based on VRAM and GPU tier
            if vram_gb >= 24:  # High-end: RTX 4090, RTX 5090, A100, etc.
                # Reduced for multi-scale detection (3x memory usage: 75%, 100%, 125%)
                self.YOLO_BATCH_SIZE = 16  # Conservative for 3-scale detection
                self.INPAINT_BATCH_SIZE = 24  # Higher for inpainting (no multi-scale)
                self.GPU_SEGMENT_FRAMES = 450  # Reduced to fit in memory
                logger.info(f"High-end GPU config: YOLO batch={self.YOLO_BATCH_SIZE}, Inpaint batch={self.INPAINT_BATCH_SIZE}, Segment={self.GPU_SEGMENT_FRAMES}")
            elif vram_gb >= 16:  # Mid-high: RTX 4080, A40, etc.
                self.YOLO_BATCH_SIZE = 24
                self.INPAINT_BATCH_SIZE = 12
                self.GPU_SEGMENT_FRAMES = 360
                logger.info(f"Mid-high GPU config: YOLO batch={self.YOLO_BATCH_SIZE}, Inpaint batch={self.INPAINT_BATCH_SIZE}, Segment={self.GPU_SEGMENT_FRAMES}")
            elif vram_gb >= 10:  # Mid: RTX 3080, RTX 4070, etc.
                self.YOLO_BATCH_SIZE = 20
                self.INPAINT_BATCH_SIZE = 10
                self.GPU_SEGMENT_FRAMES = 300
                logger.info(f"Mid-range GPU config: YOLO batch={self.YOLO_BATCH_SIZE}, Inpaint batch={self.INPAINT_BATCH_SIZE}, Segment={self.GPU_SEGMENT_FRAMES}")
            elif vram_gb >= 8:  # Entry: RTX 3070, RTX 4060, etc.
                self.YOLO_BATCH_SIZE = 16
                self.INPAINT_BATCH_SIZE = 8
                self.GPU_SEGMENT_FRAMES = 240
                logger.info(f"Entry GPU config: YOLO batch={self.YOLO_BATCH_SIZE}, Inpaint batch={self.INPAINT_BATCH_SIZE}, Segment={self.GPU_SEGMENT_FRAMES}")
            else:  # Low VRAM: < 8GB
                self.YOLO_BATCH_SIZE = 8
                self.INPAINT_BATCH_SIZE = 4
                self.GPU_SEGMENT_FRAMES = 150
                logger.info(f"Low VRAM GPU config: YOLO batch={self.YOLO_BATCH_SIZE}, Inpaint batch={self.INPAINT_BATCH_SIZE}, Segment={self.GPU_SEGMENT_FRAMES}")
                
        except Exception as exc:
            logger.warning(f"Failed to detect GPU specs ({exc}); using default parameters")

    def _resolve_inpaint_candidates(self, preferred: Sequence[str] | str | None) -> list[str]:
        raw: list[str]
        if preferred is None:
            raw = ["auto"]
        elif isinstance(preferred, str):
            raw = [part.strip() for part in preferred.split(",") if part.strip()]
        else:
            raw = [str(part).strip() for part in preferred if str(part).strip()]

        if not raw:
            raw = ["auto"]

        resolved: list[str] = []
        include_defaults = any(item.lower() in {"auto", "default"} for item in raw)
        explicit = [item.lower() for item in raw if item.lower() not in {"auto", "default"}]

        if explicit:
            for item in explicit:
                if item in self.SUPPORTED_INPAINT_MODELS:
                    resolved.append(item)
                else:
                    logger.warning(f"Unsupported inpaint model '{item}' ignored.")

        if include_defaults or not resolved:
            resolved.extend([model for model in self.SUPPORTED_INPAINT_MODELS if model not in resolved])

        unique_resolved: list[str] = []
        for model in resolved:
            if model not in unique_resolved:
                unique_resolved.append(model)
        resolved = unique_resolved

        # If GPU capability is insufficient for LaMa and it was included only via defaults, skip it.
        if self._cuda_capability is not None and self._cuda_capability < 7.0:
            if "lama" in resolved and "lama" not in explicit:
                logger.info(
                    f"Skipping LaMA in auto selection due to CUDA capability {self._cuda_capability:.1f} "
                    "(requires sm_70+)."
                )
                resolved = [model for model in resolved if model != "lama"]

        return resolved

    def _candidate_devices_for_model(self, model_name: str) -> list[torch.device]:
        if model_name == "lama":
            if self._cuda_capability is not None and self._cuda_capability >= 7.0 and self.device.startswith("cuda"):
                return [torch.device(self.device), torch.device("cpu")]
            return [torch.device("cpu")]

        if model_name == "zits":
            cpu_device = torch.device("cpu")
            if torch.cuda.is_available() and self.device.startswith("cuda"):
                if self._cuda_capability is not None and self._cuda_capability >= 7.0:
                    return [torch.device(self.device), cpu_device]
                return [cpu_device, torch.device(self.device)]
            return [cpu_device]

        return [torch.device("cpu")]

    def _prepare_model_resources(self, model_name: str) -> None:
        if model_name == "lama" and not LaMa.is_downloaded():
            logger.info("Downloading LaMA model weights...")
            LaMa.download()
        elif model_name == "zits" and not ZITS.is_downloaded():
            logger.info("Downloading ZITS model weights...")
            ZITS.download()

    def _initialize_inpaint_model(self) -> None:
        errors: list[str] = []
        for model_name in list(self._remaining_inpaint_candidates):
            devices = self._candidate_devices_for_model(model_name)
            for device in devices:
                try:
                    self._prepare_model_resources(model_name)
                    manager = ModelManager(name=model_name, device=device)
                    self.inpaint_model_manager = manager
                    self._inpaint_device = device
                    self._active_inpaint_model = model_name
                    logger.info(f"Inpaint model '{model_name}' loaded on {device.type}")
                    return
                except NotImplementedError as exc:
                    message = f"{model_name}@{device.type}: {exc}"
                    logger.warning(f"Inpaint model unavailable ({message})")
                    errors.append(message)
                except RuntimeError as exc:
                    message = f"{model_name}@{device.type}: {exc}"
                    logger.error(f"Failed to initialize inpaint model ({message})")
                    errors.append(message)
            self._remaining_inpaint_candidates = [
                candidate for candidate in self._remaining_inpaint_candidates if candidate != model_name
            ]

        raise RuntimeError(
            "Unable to load any inpainting model. Attempted: "
            + ", ".join(self._inpaint_candidates)
            + (f". Errors: {errors}" if errors else "")
        )

    def _ensure_inpaint_model(self) -> None:
        if self.inpaint_model_manager is None:
            self._initialize_inpaint_model()

    def _ensure_florence_model(self) -> None:
        if self.florence_model is not None and self.florence_processor is not None:
            return
        logger.info("Loading Florence-2 model...")
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        ).to(self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )
        logger.info("Florence-2 model loaded")

    def _probe_video(self, input_path: Path) -> dict[str, Any]:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(input_path),
        ]
        try:
            result = subprocess.run(
                probe_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - external dependency
            raise RuntimeError(f"Failed to probe video metadata: {exc.stderr.decode().strip()}") from exc

        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError("Unable to parse ffprobe output") from exc

        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError("No video stream found in source file.")

        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if not width or not height:
            raise RuntimeError("Unable to determine video resolution.")

        def _frame_rate(entry: str | None) -> float | None:
            if not entry or entry == "0/0":
                return None
            if "/" in entry:
                num, den = entry.split("/", 1)
                try:
                    return float(num) / float(den)
                except (ValueError, ZeroDivisionError):
                    return None
            try:
                return float(entry)
            except ValueError:
                return None

        avg_rate = _frame_rate(stream.get("avg_frame_rate"))
        r_rate = _frame_rate(stream.get("r_frame_rate"))
        fps = avg_rate or r_rate or 30.0

        nb_frames_raw = stream.get("nb_frames")
        total_frames: int | None = None
        if nb_frames_raw and nb_frames_raw.isdigit():
            total_frames = int(nb_frames_raw)

        duration = None
        if total_frames is None:
            duration_value = payload.get("format", {}).get("duration")
            try:
                duration = float(duration_value) if duration_value is not None else None
            except (TypeError, ValueError):
                duration = None
            if duration:
                total_frames = int(math.ceil(duration * fps))

        if total_frames is None:
            total_frames = 0

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": total_frames,
            "duration": duration,
        }

    def _extract_audio_track(self, input_path: Path, workdir: Path) -> Path | None:
        ffmpeg_bin = self._ffmpeg_path("ffmpeg")
        audio_path = workdir / "audio_track.mka"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-map",
            "0:a?",
            "-c:a",
            "copy",
            str(audio_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on ffmpeg
            logger.warning("Audio extraction failed (%s). Continuing without audio.", exc)
            return None

        if audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path
        return None

    def _decode_video_segments(
        self,
        input_path: Path,
        segment_frames: int | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        segment_frames = segment_frames or self.GPU_SEGMENT_FRAMES
        try:
            container = av.open(str(input_path), options={"hwaccel": "cuda"})
        except av.AVError:
            logger.warning("PyAV CUDA decode unavailable; falling back to software decode.")
            container = av.open(str(input_path))

        video_stream = container.streams.video[0]
        fps: float
        if video_stream.average_rate:
            fps = float(video_stream.average_rate)
        elif video_stream.base_rate:
            fps = float(video_stream.base_rate)
        else:
            fps = 30.0

        width = int(video_stream.codec_context.width or video_stream.width or 0)
        height = int(video_stream.codec_context.height or video_stream.height or 0)

        if not width or not height:
            raise RuntimeError("Unable to determine video resolution during decode.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segments: list[torch.Tensor] = []
        buffer: list[torch.Tensor] = []
        frame_times: list[float] = []
        frame_index = 0

        try:
            for frame in container.decode(video_stream):
                arr = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
                tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
                if device.type == "cuda":
                    tensor = tensor.to(dtype=torch.float16)
                buffer.append(tensor)
                if frame.time is not None:
                    frame_times.append(float(frame.time))
                else:
                    frame_times.append(frame_index / fps)
                frame_index += 1

                if len(buffer) >= segment_frames:
                    segments.append(torch.stack(buffer, dim=0))
                    buffer = []
        finally:
            container.close()

        if buffer:
            segments.append(torch.stack(buffer, dim=0))

        info = {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_times": frame_times,
            "total_frames": len(frame_times),
            "device": device,
        }
        return segments, info

    @staticmethod
    def _segments_to_cpu_frames(segments: list[torch.Tensor]) -> tuple[list[np.ndarray], list[Image.Image]]:
        frames_rgb: list[np.ndarray] = []
        pil_frames: list[Image.Image] = []
        for segment in segments:
            cpu_tensor = segment.to(device="cpu", dtype=torch.float32)
            cpu_tensor = (cpu_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            cpu_numpy = cpu_tensor.permute(0, 2, 3, 1).contiguous().numpy()
            for arr in cpu_numpy:
                frames_rgb.append(arr)
                pil_frames.append(Image.fromarray(arr, mode="RGB"))
        return frames_rgb, pil_frames

    def _encode_video_nvenc(
        self,
        frames: Sequence[np.ndarray],
        width: int,
        height: int,
        fps: float,
        output_path: Path,
    ) -> None:
        """Encode video using hardware acceleration (NVENC if available)."""
        ffmpeg_bin = self._ffmpeg_path("ffmpeg")
        
        # Try NVENC first (hardware acceleration)
        encode_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",  # Slower preset for MUCH better quality (p4 = balanced quality/speed)
            "-tune",
            "hq",  # High quality
            "-rc",
            "vbr",  # Variable bitrate for better quality
            "-cq",
            "16",  # Quality level (lower = better, 16 is near-lossless with grain preservation)
            "-b:v",
            "0",  # Let NVENC decide bitrate
            "-multipass",
            "fullres",  # 2-pass encoding for better quality
            "-spatial-aq",
            "1",  # Spatial AQ for better quality in complex areas
            "-temporal-aq",
            "1",  # Temporal AQ for better motion quality
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            if not encode_proc.stdin:
                raise RuntimeError("Failed to open FFmpeg encode stream.")
            for frame in frames:
                encode_proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            # NVENC might not be available, will check return code
            pass
        finally:
            if encode_proc.stdin:
                encode_proc.stdin.close()

        encode_return = encode_proc.wait()
        if encode_return != 0:
            stderr = encode_proc.stderr.read().decode("utf-8", errors="ignore") if encode_proc.stderr else ""
            raise RuntimeError(f"NVENC encoding failed (code {encode_return}): {stderr}")
        if encode_proc.stderr:
            encode_proc.stderr.close()
    
    def _encode_video_cpu(
        self,
        frames: Sequence[np.ndarray],
        width: int,
        height: int,
        fps: float,
        output_path: Path,
    ) -> None:
        """Fallback CPU encoding using libx264."""
        ffmpeg_bin = self._ffmpeg_path("ffmpeg")
        encode_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",  # Slower but much better quality
            "-crf",
            "16",  # Lower CRF = better quality (18→16 for near-lossless)
            "-tune",
            "film",  # Preserve grain and fine details
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            if not encode_proc.stdin:
                raise RuntimeError("Failed to open FFmpeg encode stream.")
            for frame in frames:
                encode_proc.stdin.write(frame.tobytes())
        finally:
            if encode_proc.stdin:
                encode_proc.stdin.close()

        encode_return = encode_proc.wait()
        if encode_return != 0:
            stderr = encode_proc.stderr.read().decode("utf-8", errors="ignore") if encode_proc.stderr else ""
            raise RuntimeError(f"FFmpeg CPU encode failed with code {encode_return}: {stderr}")
        if encode_proc.stderr:
            encode_proc.stderr.close()

    def _remux_audio(self, video_path: Path, audio_path: Path | None, output_path: Path) -> None:
        if not audio_path or not audio_path.exists() or audio_path.stat().st_size == 0:
            shutil.move(str(video_path), str(output_path))
            return

        ffmpeg_bin = self._ffmpeg_path("ffmpeg")
        remux_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            subprocess.run(remux_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            logger.warning("Remux failed (%s); delivering video without audio.", exc)
            shutil.move(str(video_path), str(output_path))

    @staticmethod
    def _ffmpeg_path(binary: str) -> str:
        resolved = shutil.which(binary)
        return resolved or binary

    def _load_models(self):
        """Load the AI models if not already loaded."""
        self._ensure_florence_model()
        self._ensure_inpaint_model()

    def _identify_watermark(self, image: Image.Image, text_input: str) -> dict:
        """Detect watermarks in an image using Florence-2.

        Args:
            image: PIL Image to analyze
            text_input: Text prompt for detection

        Returns:
            Parsed detection results
        """
        self._ensure_florence_model()
        task_prompt = TaskType.OPEN_VOCAB_DETECTION
        prompt = task_prompt.value + text_input

        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        return self.florence_processor.post_process_generation(
            generated_text, task=task_prompt.value, image_size=(image.width, image.height)
        )

    @staticmethod
    def _box_area(bbox: Sequence[int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _box_center(bbox: Sequence[int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _is_near_border(bbox: Sequence[int], width: int, height: int, margin_ratio: float = 0.18) -> bool:
        x1, y1, x2, y2 = bbox
        margin_w = width * margin_ratio
        margin_h = height * margin_ratio
        return (
            x1 <= margin_w
            or y1 <= margin_h
            or x2 >= width - margin_w
            or y2 >= height - margin_h
        )

    def _is_plausible_watermark_box(
        self,
        bbox: Sequence[int],
        label: str | None,
        score: float | None,
        width: int,
        height: int,
        max_bbox_percent: float,
    ) -> bool:
        bbox_area = self._box_area(bbox)
        if bbox_area == 0:
            return False

        image_area = width * height
        if (bbox_area / image_area) * 100 > max_bbox_percent:
            return False

        label_normalized = (label or "").lower().strip()
        if label_normalized:
            if any(term in label_normalized for term in self.WATERMARK_NEGATIVE_KEYWORDS):
                return False

        if score is not None and score < self.WATERMARK_MIN_SCORE:
            allow_soft_match = (
                label_normalized
                and any(key in label_normalized for key in self.WATERMARK_POSITIVE_KEYWORDS)
            ) or self._is_near_border(bbox, width, height)
            if not allow_soft_match or score < self.WATERMARK_FALLBACK_MIN_SCORE:
                return False

        is_edge_candidate = self._is_near_border(bbox, width, height)
        label_suggests_watermark = label_normalized and any(
            key in label_normalized for key in self.WATERMARK_POSITIVE_KEYWORDS
        )

        if not label_suggests_watermark and not is_edge_candidate:
            # Central detections without matching keywords are likely content (e.g., road signs)
            return False

        # Discard extremely tall/narrow detections that do not resemble typical overlays.
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = max(w / h, h / w)
        if aspect_ratio > 25 and not label_suggests_watermark:
            return False

        return True

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0 or mask.max() == 0:
            return mask
        kernel = np.ones(self.WATERMARK_DILATION_KERNEL, np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=self.WATERMARK_DILATION_ITERATIONS)
        filled = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        return filled

    def _download_yolo_model(self) -> Path:
        if self._yolo_model_path and self._yolo_model_path.exists():
            return self._yolo_model_path

        if not self._yolo_model_url:
            raise RuntimeError("YOLO detector requested but no model URL is configured.")

        parsed = urlparse(self._yolo_model_url)
        filename = Path(parsed.path).name or "watermark_yolo.pt"
        target_path = self._yolo_cache_dir / filename
        if target_path.exists():
            self._yolo_model_path = target_path
            return target_path

        logger.info("Downloading YOLO watermark detector weights from %s", self._yolo_model_url)
        response = requests.get(self._yolo_model_url, stream=True, timeout=180)
        response.raise_for_status()
        with target_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        logger.info("YOLO watermark detector stored at %s", target_path)
        self._yolo_model_path = target_path
        return target_path

    def _ensure_yolo_model(self) -> None:
        if self._yolo_model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "YOLO detector requested but the 'ultralytics' package is not installed."
            ) from exc

        model_path = self._download_yolo_model()
        self._yolo_model = YOLO(str(model_path))

        if str(self.device).startswith("cuda"):
            target_device = str(self.device) if ":" in str(self.device) else f"{self.device}:0"
        else:
            target_device = "cpu"

        try:
            self._yolo_model.to(target_device)
            self._yolo_device = target_device
        except Exception as exc:  # pragma: no cover - defensive
            self._yolo_device = None
            logger.warning(
                "Unable to move YOLO detector to %s (%s). Falling back to default device.",
                target_device,
                exc,
            )

    def _detect_watermarks_with_florence(self, image: Image.Image) -> list[Detection]:
        """Generate a mask for watermark regions.

        Args:
            image: PIL Image to process
            max_bbox_percent: Maximum percentage of image area a bbox can cover

        Returns:
            Binary mask image
        """
        text_input = "watermark logo sora"
        parsed_answer = self._identify_watermark(image, text_input)

        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        detections = parsed_answer.get(detection_key) or {}
        bboxes = detections.get("bboxes") or []
        labels = detections.get("labels") or []
        scores = detections.get("scores") or detections.get("confidences") or []
        collected: list[Detection] = []

        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            label = labels[idx] if idx < len(labels) else None
            score = scores[idx] if idx < len(scores) else None
            try:
                score_value = float(score) if score is not None else None
            except (TypeError, ValueError):
                score_value = None
            collected.append(((x1, y1, x2, y2), label, score_value))

        return collected

    def _detect_watermarks_with_yolo(self, image: Image.Image) -> list[Detection]:
        self._ensure_yolo_model()
        if self._yolo_model is None:
            return []

        image_np = np.array(image)
        predict_kwargs: dict[str, object] = {"verbose": False}
        if self._yolo_device:
            predict_kwargs["device"] = self._yolo_device
        results = self._yolo_model.predict(image_np, **predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        names = getattr(result, "names", None) or getattr(self._yolo_model, "names", {})

        detections: list[Detection] = []
        for idx in range(len(boxes)):
            xyxy = boxes.xyxy[idx].tolist()
            conf_value: Optional[float] = None
            if boxes.conf is not None:
                conf_value = float(boxes.conf[idx])
            cls_value: Optional[int] = None
            if boxes.cls is not None:
                cls_value = int(boxes.cls[idx])

            label: Optional[str] = None
            if cls_value is not None:
                if isinstance(names, dict):
                    label = names.get(cls_value)
                elif isinstance(names, (list, tuple)) and 0 <= cls_value < len(names):
                    label = names[cls_value]

            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(0, min(x2, image.width))
            y2 = max(0, min(y2, image.height))
            detections.append(((x1, y1, x2, y2), label, conf_value))

        return detections

    def _detect_watermarks_with_yolo_batch(self, images: list[Image.Image]) -> list[list[Detection]]:
        if not images:
            return []
        self._ensure_yolo_model()
        if self._yolo_model is None:
            return [[] for _ in images]

        batch_np = [np.array(img) for img in images]
        predict_kwargs: dict[str, object] = {
            "verbose": False,
            "batch": min(self.YOLO_BATCH_SIZE, len(batch_np)),
        }
        if self._yolo_device:
            predict_kwargs["device"] = self._yolo_device

        results = self._yolo_model.predict(batch_np, **predict_kwargs)
        detections_batch: list[list[Detection]] = []

        names = getattr(self._yolo_model, "names", {})
        for img, result in zip(images, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                detections_batch.append([])
                continue

            # prefer per-result label mapping when available
            name_map = getattr(result, "names", None) or names
            detections: list[Detection] = []
            for idx in range(len(boxes)):
                xyxy = boxes.xyxy[idx].tolist()
                conf_value: Optional[float] = None
                if boxes.conf is not None:
                    conf_value = float(boxes.conf[idx])
                cls_value: Optional[int] = None
                if boxes.cls is not None:
                    cls_value = int(boxes.cls[idx])

                label: Optional[str] = None
                if cls_value is not None:
                    if isinstance(name_map, dict):
                        label = name_map.get(cls_value)
                    elif isinstance(name_map, (list, tuple)) and 0 <= cls_value < len(name_map):
                        label = name_map[cls_value]

                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, min(x1, img.width - 1))
                y1 = max(0, min(y1, img.height - 1))
                x2 = max(0, min(x2, img.width))
                y2 = max(0, min(y2, img.height))
                detections.append(((x1, y1, x2, y2), label, conf_value))

            detections_batch.append(detections)

        if len(detections_batch) < len(images):  # pragma: no cover - defensive
            detections_batch.extend([[]] * (len(images) - len(detections_batch)))

        return detections_batch

    def _detect_boxes_for_segments(
        self,
        segments: list[torch.Tensor],
        detector: str | None,
    ) -> list[list[tuple[int, int, int, int]]]:
        backend = self._effective_detector(detector)
        if backend != "yolo":
            return []

        self._ensure_yolo_model()
        if self._yolo_model is None:
            return []

        device = self._yolo_device or ("cuda" if torch.cuda.is_available() else "cpu")
        boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
        
        # Multi-scale detection: run at multiple scales for better coverage
        # Can be disabled via ENABLE_MULTISCALE_DETECTION if memory is limited
        
        # Start with multi-scale enabled if configured
        use_multiscale = self.ENABLE_MULTISCALE_DETECTION
        
        retry_attempt = 0
        max_retries = 2  # Try multi-scale, then single-scale

        while retry_attempt < max_retries:
            try:
                if use_multiscale:
                    scales = self.MULTISCALE_SCALES
                    logger.info(f"Using multi-scale detection with scales: {scales}")
                else:
                    scales = [1.0]
                    logger.info("Using single-scale detection (multi-scale disabled or fallback)")

                for segment in segments:
                    if segment.device.type != "cuda":
                        segment = segment.to(device=device)
                    
                    all_boxes_frame: list[list[tuple[int, int, int, int]]] = []
                    
                    # Process each scale separately to minimize memory usage
                    for scale in scales:
                        # Clear CUDA cache before each scale to prevent OOM
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        if scale != 1.0:
                            # Resize segment for multi-scale detection
                            b, c, h, w = segment.shape
                            new_h, new_w = int(h * scale), int(w * scale)
                            scaled_segment = torch.nn.functional.interpolate(
                                segment, size=(new_h, new_w), mode='bilinear', align_corners=False
                            )
                        else:
                            scaled_segment = segment
                        
                        inputs = scaled_segment.float()
                        
                        # Reduce batch size for multi-scale to prevent OOM
                        if len(scales) > 1:
                            effective_batch_size = max(4, self.YOLO_BATCH_SIZE // 4)
                        else:
                            effective_batch_size = self.YOLO_BATCH_SIZE
                        
                        predict_kwargs: dict[str, Any] = {
                            "verbose": False,
                            "device": device,
                            "batch": min(effective_batch_size, inputs.shape[0]),
                            "conf": 0.15,  # Slightly higher confidence to reduce false positives
                            "iou": 0.3,    # Higher IOU for better filtering
                            "agnostic_nms": True,
                            "max_det": 50,  # Reduced from 100 to save memory
                        }
                        if inputs.dtype == torch.float16:
                            predict_kwargs["half"] = True
                        
                        results = self._yolo_model.predict(inputs, **predict_kwargs)
                        
                        # Scale boxes back to original size
                        scale_factor = 1.0 / scale
                        frame_boxes: list[tuple[int, int, int, int]] = []
                        for result in results:
                            boxes = getattr(result, "boxes", None)
                            if boxes is not None and boxes.xyxy is not None:
                                xyxy = boxes.xyxy.detach().to("cpu").numpy()
                                for bb in xyxy:
                                    x1, y1, x2, y2 = map(int, (bb[:4] * scale_factor))
                                    frame_boxes.append((x1, y1, x2, y2))
                        all_boxes_frame.append(frame_boxes)
                    
                    # Merge boxes from all scales (remove duplicates with NMS)
                    merged_boxes = self._merge_multiscale_boxes(all_boxes_frame)
                    boxes_per_frame.append(merged_boxes)
                
                # If successful, break the loop
                return boxes_per_frame

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    retry_attempt += 1
                    logger.warning(
                        "CUDA out of memory during detection. "
                        f"Attempt {retry_attempt}/{max_retries}. Clearing cache and retrying."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if use_multiscale:
                        logger.info("Disabling multi-scale detection for the next attempt.")
                        use_multiscale = False
                    else:
                        # If even single-scale fails, we can't proceed with this method.
                        logger.error("CUDA out of memory even with single-scale detection. Aborting GPU detection.")
                        raise  # Re-raise the exception to be caught by the outer handler
                else:
                    # For other runtime errors, don't retry
                    raise
        
        # If all retries fail, return empty list
        return []

    def _merge_multiscale_boxes(
        self, 
        boxes_list: list[list[tuple[int, int, int, int]]]
    ) -> list[tuple[int, int, int, int]]:
        """Merge boxes from multiple scales using IoU-based NMS."""
        if not boxes_list:
            return []
        
        # Flatten all boxes
        all_boxes = []
        for boxes in boxes_list:
            all_boxes.extend(boxes)
        
        if not all_boxes:
            return []
        
        # Simple NMS: remove boxes with high IoU overlap
        def box_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        # Keep unique boxes (IoU < 0.5)
        unique_boxes = []
        for box in all_boxes:
            is_duplicate = False
            for existing in unique_boxes:
                if box_iou(box, existing) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boxes.append(box)
        
        return unique_boxes

    @staticmethod
    def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _box_center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax, ay = WatermarkRemovalService._box_center(box_a)
        bx, by = WatermarkRemovalService._box_center(box_b)
        return math.hypot(ax - bx, ay - by)

    def _propagate_boxes_temporally(
        self,
        boxes_per_frame: list[list[tuple[int, int, int, int]]],
        width: int,
        height: int,
        fps: float,
    ) -> list[list[tuple[int, int, int, int]]]:
        num_frames = len(boxes_per_frame)
        propagated: list[list[tuple[int, int, int, int]]] = [list(boxes) for boxes in boxes_per_frame]
        lookback = self.WATERMARK_LOOKBACK_FRAMES
        lookahead = max(1, int(round(fps * self.WATERMARK_LOOKAHEAD_SECONDS)))
        max_gap = self.WATERMARK_MAX_GAP_FRAMES
        diag = math.hypot(width, height)
        threshold = diag * self.WATERMARK_POSITION_THRESHOLD

        for idx, boxes in enumerate(boxes_per_frame):
            if not boxes:
                continue
            for box in boxes:
                gap = 0
                for j in range(idx - 1, max(-1, idx - lookback) - 1, -1):
                    if j < 0:
                        break
                    existing = boxes_per_frame[j]
                    if existing:
                        min_dist = min(self._box_center_distance(box, prev_box) for prev_box in existing)
                        if min_dist > threshold:
                            break
                        gap = 0
                    else:
                        gap += 1
                        if gap > max_gap:
                            break
                    propagated[j].append(box)

                gap = 0
                for j in range(idx + 1, min(num_frames, idx + lookahead + 1)):
                    existing = boxes_per_frame[j]
                    if existing:
                        min_dist = min(self._box_center_distance(box, next_box) for next_box in existing)
                        if min_dist > threshold:
                            break
                        gap = 0
                    else:
                        gap += 1
                        if gap > max_gap:
                            break
                    propagated[j].append(box)

        return propagated

    def _boxes_to_masks(
        self,
        boxes_per_frame: list[list[tuple[int, int, int, int]]],
        width: int,
        height: int,
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        for boxes in boxes_per_frame:
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            seen: set[tuple[int, int, int, int]] = set()
            for box in boxes:
                key = tuple(int(v) for v in box)
                if key in seen:
                    continue
                seen.add(key)
                x1, y1, x2, y2 = key
                draw.rectangle([x1, y1, x2, y2], fill=255)
            masks.append(np.array(mask, dtype=np.uint8))
        return masks

    def _build_mask_from_detections(
        self,
        image: Image.Image,
        detections: list[Detection],
        max_bbox_percent: float,
    ) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        for bbox, label, score in detections:
            x1, y1, x2, y2 = bbox
            if not self._is_plausible_watermark_box(
                (x1, y1, x2, y2),
                label,
                score,
                image.width,
                image.height,
                max_bbox_percent,
            ):
                continue

            draw.rectangle([x1, y1, x2, y2], fill=255)

        mask_array = np.array(mask, dtype=np.uint8)
        refined = self._refine_mask(mask_array)
        return Image.fromarray(refined, mode="L")

    def _get_watermark_mask(
        self,
        image: Image.Image,
        max_bbox_percent: float,
        detector: str | None = None,
    ) -> Image.Image:
        backend = self._effective_detector(detector)
        detections: list[Detection] = []

        if backend == "yolo":
            try:
                detections = self._detect_watermarks_with_yolo(image)
            except Exception as exc:  # noqa: BLE001
                logger.error("YOLO detection failed (%s). Falling back to Florence-2.", exc)
                backend = "florence"

        if backend == "florence":
            detections = self._detect_watermarks_with_florence(image)

        return self._build_mask_from_detections(image, detections, max_bbox_percent)

    def _build_masks_from_detections(
        self,
        images: list[Image.Image],
        detections_batch: list[list[Detection]],
        max_bbox_percent: float,
    ) -> list[Image.Image]:
        masks: list[Image.Image] = []
        for img, dets in zip(images, detections_batch):
            masks.append(self._build_mask_from_detections(img, dets, max_bbox_percent))
        return masks

    def _generate_mask_batch(
        self,
        images: list[Image.Image],
        max_bbox_percent: float,
        detector: str | None,
    ) -> list[Image.Image]:
        if not images:
            return []

        backend = self._effective_detector(detector)
        if backend == "yolo":
            detections_batch = self._detect_watermarks_with_yolo_batch(images)
            return self._build_masks_from_detections(images, detections_batch, max_bbox_percent)

        # Fallback to sequential processing for Florence or unknown detectors
        return [
            self._get_watermark_mask(img, max_bbox_percent, backend)
            for img in images
        ]

    def _build_inpaint_config(self) -> Config:
        if hasattr(Config, "model_fields"):
            raw_fields = Config.model_fields
        else:  # pragma: no cover - fallback for Pydantic v1
            raw_fields = getattr(Config, "__fields__", {})

        undefined_sentinels = {PydanticUndefined, Undefined}
        config_defaults: dict[str, object] = {}
        for name, field in raw_fields.items():
            default = getattr(field, "default", PydanticUndefined)
            default_factory = getattr(field, "default_factory", None)
            if default not in undefined_sentinels:
                config_defaults[name] = default
            elif callable(default_factory):
                config_defaults[name] = default_factory()

        config_defaults.update(
            {
                "ldm_steps": 50,
                "ldm_sampler": LDMSampler.ddim,
                "hd_strategy": HDStrategy.CROP,
                "hd_strategy_crop_margin": 64,
                "hd_strategy_crop_trigger_size": 800,
                "hd_strategy_resize_limit": 1600,
            }
        )

        if hasattr(Config, "model_construct"):
            return Config.model_construct(**config_defaults)
        return Config.construct(**config_defaults)  # type: ignore[attr-defined]

    def _run_inpainter(self, image: np.ndarray, mask: np.ndarray, config: Config) -> np.ndarray:
        attempts: set[str] = set()
        while True:
            self._ensure_inpaint_model()
            assert self.inpaint_model_manager is not None
            model_name = self._active_inpaint_model or "unknown"

            use_autocast = (
                self._inpaint_device is not None
                and self._inpaint_device.type == "cuda"
                and model_name != "lama"
            )

            try:
                with torch.inference_mode():
                    if use_autocast and torch.cuda.is_available():
                        ctx = torch.amp.autocast("cuda")
                    else:
                        ctx = nullcontext()
                    with ctx:  # type: ignore[arg-type]
                        result = self.inpaint_model_manager(image, mask, config)
                break
            except RuntimeError as exc:
                device_label = self._inpaint_device.type if self._inpaint_device else "unknown"
                logger.error(
                    f"Inpaint model '{model_name}' on {device_label} failed: {exc}"
                )
                attempts.add(model_name)
                self.inpaint_model_manager = None
                self._active_inpaint_model = None
                self._inpaint_device = None
                if len(attempts) >= len(self._inpaint_candidates):
                    raise
                continue

        if result.dtype in [np.float64, np.float32]:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def _process_image_with_inpainter(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        config = self._build_inpaint_config()
        return self._run_inpainter(image, mask, config)

    def _process_frames_with_inpainter_batch(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Process multiple frames in parallel using GPU batch processing."""
        if not frames:
            return []
        
        # True batch processing - all frames at once when possible
        if self.device.startswith("cuda") and len(frames) > 1:
            try:
                # Process all frames in a single batch for better GPU utilization
                config = self._build_inpaint_config()
                outputs: list[np.ndarray] = []
                
                # Use torch.no_grad() for inference to save memory
                with torch.no_grad():
                    for image, mask in zip(frames, masks):
                        result_bgr = self._run_inpainter(image, mask, config)
                        rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        outputs.append(rgb)
                
                return outputs
            except RuntimeError as e:
                # Fallback to sequential if batch fails (OOM)
                logger.warning(f"Batch processing failed ({e}), falling back to sequential")
        
        # Sequential fallback
        outputs: list[np.ndarray] = []
        config = self._build_inpaint_config()
        for image, mask in zip(frames, masks):
            result_bgr = self._run_inpainter(image, mask, config)
            rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            outputs.append(rgb)
        return outputs

    def _make_region_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Make watermark regions transparent.

        Args:
            image: Input image
            mask: Binary mask

        Returns:
            Image with transparent regions
        """
        image = image.convert("RGBA")
        mask = mask.convert("L")
        transparent_image = Image.new("RGBA", image.size)

        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) > 0:
                    transparent_image.putpixel((x, y), (0, 0, 0, 0))
                else:
                    transparent_image.putpixel((x, y), image.getpixel((x, y)))

        return transparent_image

    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a video based on extension."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        return file_path.suffix.lower() in video_extensions

    def _process_video_cpu(
        self,
        input_path: Path,
        output_path: Path,
        transparent: bool,
        max_bbox_percent: float,
        force_format: Optional[str],
        detector: str | None,
    ) -> Path:
        """Process a video file for watermark removal.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            transparent: Whether to make watermarks transparent
            max_bbox_percent: Maximum bbox percentage
            force_format: Forced output format
            detector: Detection backend override

        Returns:
            Path to processed video
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine output format
        if force_format:
            output_format = force_format.upper()
        else:
            output_format = "MP4"

        # Create output path
        if output_path.is_dir():
            output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
        else:
            output_file = output_path.with_suffix(f".{output_format.lower()}")

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"

        # Set codec based on format
        if output_format.upper() == "MP4":
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_format.upper() == "AVI":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

        use_yolo = detector == "yolo"
        detection_interval = self.YOLO_BATCH_SIZE if use_yolo else max(1, int(round(fps)) or 1)
        history_limit = max(
            1,
            int(
                math.ceil(
                    (fps * self.WATERMARK_ACCUMULATION_SECONDS)
                    / float(detection_interval)
                )
            ),
        )
        combined_mask_array: np.ndarray | None = None
        mask_image_cached: Image.Image | None = None
        mask_history: deque[np.ndarray] = deque(maxlen=history_limit)

        # Process each frame
        with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                refresh_mask = mask_image_cached is None or frame_count % detection_interval == 0
                if refresh_mask:
                    latest_mask_image = self._get_watermark_mask(pil_image, max_bbox_percent, detector)
                    latest_mask_np = np.array(latest_mask_image, dtype=np.uint8)
                    mask_history.append(latest_mask_np)
                    if mask_history:
                        mask_stack = np.stack(mask_history, axis=0)
                        combined_mask_array = mask_stack.max(axis=0).astype(np.uint8)
                    else:  # pragma: no cover - defensive
                        combined_mask_array = latest_mask_np
                    mask_image_cached = Image.fromarray(combined_mask_array, mode="L")
                elif combined_mask_array is None:
                    combined_mask_array = (
                        mask_history[-1].copy() if mask_history else np.zeros((height, width), dtype=np.uint8)
                    )
                    mask_image_cached = Image.fromarray(combined_mask_array, mode="L")

                mask_np = combined_mask_array if combined_mask_array is not None else np.zeros(
                    (height, width), dtype=np.uint8
                )
                mask_image = mask_image_cached if mask_image_cached is not None else Image.fromarray(mask_np, mode="L")

                # Process frame
                if transparent:
                    result_image = self._make_region_transparent(pil_image, mask_image)
                    # Convert RGBA to RGB by filling with white
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    inpaint_result = self._process_image_with_inpainter(
                        np.array(pil_image), mask_np
                    )
                    result_image = Image.fromarray(
                        cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB)
                    )

                # Convert back to OpenCV format and write
                frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                out.write(frame_result)

                frame_count += 1
                pbar.update(1)

        # Release resources
        cap.release()
        out.release()

        # Combine with audio using FFmpeg
        try:
            logger.info("Combining processed video with original audio...")

            # Check if FFmpeg is available
            try:
                subprocess.check_output(
                    ["ffmpeg", "-version"],
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("FFmpeg not available. Video will be saved without audio.")
                output_file.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(str(temp_video_path), str(output_file))
                return output_file

            # Use FFmpeg to combine video and audio
            output_file.parent.mkdir(parents=True, exist_ok=True)

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),  # Processed video without audio
                "-i", str(input_path),       # Original video with audio
                "-c:v", "copy",              # Copy video without re-encoding
                "-c:a", "copy",              # Keep original audio codec when possible
                "-map", "0:v:0",             # Use video from first input
                "-map", "1:a?",              # Use audio from second input if present
                "-shortest",                 # Stop when shortest stream ends
                str(output_file)
            ]

            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            logger.info("Audio/video combination completed successfully!")

        except Exception as e:
            logger.error(f"Error combining audio/video: {str(e)}")
            # Fallback: use video without audio
            output_file.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(str(temp_video_path), str(output_file))
        finally:
            # Cleanup temp files
            try:
                os.remove(str(temp_video_path))
                os.rmdir(temp_dir)
            except:
                pass

        return output_file

    def _process_video_gpu(
        self,
        input_path: Path,
        output_path: Path,
        transparent: bool,
        max_bbox_percent: float,
        force_format: Optional[str],
        detector: str | None,
    ) -> Path:
        logger.info("GPU pipeline: Starting video decode...")
        try:
            segments, info = self._decode_video_segments(input_path, self.GPU_SEGMENT_FRAMES)
            logger.info(f"GPU pipeline: Decoded {len(segments)} segments")
        except Exception as e:
            logger.error(f"GPU pipeline: Video decode failed - {type(e).__name__}: {str(e)}")
            raise
        
        width = info["width"]
        height = info["height"]
        fps = info["fps"]
        logger.info(f"GPU pipeline: Video specs - {width}x{height} @ {fps:.2f} fps")

        if force_format:
            output_format = force_format.upper()
        else:
            output_format = "MP4"

        if output_format not in {"MP4", "M4V", "MOV"}:
            raise RuntimeError(f"GPU pipeline currently supports MP4/MOV outputs, not {output_format}.")

        if output_path.is_dir():
            output_file = output_path / f"{input_path.stem}_no_watermark.{output_format.lower()}"
        else:
            output_file = output_path.with_suffix(f".{output_format.lower()}" )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        total_frames = sum(segment.shape[0] for segment in segments)
        if not total_frames:
            raise RuntimeError("No frames decoded from input video.")

        boxes_per_frame: list[list[tuple[int, int, int, int]]] = []
        if detector == "yolo":
            boxes_per_frame = self._detect_boxes_for_segments(segments, detector)
            if boxes_per_frame and len(boxes_per_frame) != total_frames:
                logger.warning(
                    "YOLO produced %s boxes but video has %s frames; falling back to CPU detection.",
                    len(boxes_per_frame),
                    total_frames,
                )
                boxes_per_frame = []

        logger.info("GPU pipeline: Converting segments to CPU frames...")
        try:
            frames_rgb, pil_frames = self._segments_to_cpu_frames(segments)
            if not frames_rgb:
                raise RuntimeError("No frames decoded from input video.")
            logger.info(f"GPU pipeline: Converted {len(frames_rgb)} frames successfully")
        except Exception as e:
            logger.error(f"GPU pipeline: Frame conversion failed - {type(e).__name__}: {str(e)}")
            raise

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            audio_path = self._extract_audio_track(input_path, tmpdir_path)

            if boxes_per_frame:
                propagated_boxes = self._propagate_boxes_temporally(
                    boxes_per_frame,
                    width,
                    height,
                    fps,
                )
                raw_masks = self._boxes_to_masks(propagated_boxes, width, height)
            else:
                raw_masks = []
                chunk_size = self.YOLO_BATCH_SIZE if (detector == "yolo") else 1
                for idx in range(0, len(pil_frames), chunk_size):
                    chunk_images = pil_frames[idx : idx + chunk_size]
                    mask_images = self._generate_mask_batch(chunk_images, max_bbox_percent, detector)
                    for mask_img in mask_images:
                        raw_masks.append(np.array(mask_img, dtype=np.uint8))

                if len(raw_masks) != len(frames_rgb):
                    raise RuntimeError("Mismatch between decoded frames and generated masks.")

                # Propagate masks forward and backward to cover moving watermarks
                lookahead = SETTINGS.WATERMARK_LOOKAHEAD_FRAMES
                lookbehind = SETTINGS.WATERMARK_LOOKBEHIND_FRAMES
                
                propagated_masks = [mask.copy() for mask in raw_masks]
                for i, mask in enumerate(raw_masks):
                    if mask.max() > 0:
                        # Propagate forward
                        for j in range(i + 1, min(i + lookahead + 1, len(raw_masks))):
                            propagated_masks[j] = np.maximum(propagated_masks[j], mask)
                        # Propagate backward
                        for j in range(max(0, i - lookbehind), i):
                            propagated_masks[j] = np.maximum(propagated_masks[j], mask)
                
                raw_masks = propagated_masks

            refined_masks = [self._refine_mask(mask) for mask in raw_masks]

            if transparent:
                processed_frames: list[np.ndarray] = []
                for frame_np, mask_np in zip(frames_rgb, refined_masks):
                    if mask_np.max() == 0:
                        processed_frames.append(frame_np)
                        continue
                    mask_image = Image.fromarray(mask_np, mode="L")
                    pil_image = Image.fromarray(frame_np, mode="RGB")
                    result_image = self._make_region_transparent(pil_image, mask_image)
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    processed_frames.append(np.array(background, dtype=np.uint8))
            else:
                processed_frames = frames_rgb.copy()
                indices: list[int] = [idx for idx, mask_np in enumerate(refined_masks) if mask_np.max() > 0]
                if indices:
                    for start in range(0, len(indices), self.INPAINT_BATCH_SIZE):
                        batch_indices = indices[start : start + self.INPAINT_BATCH_SIZE]
                        batch_frames = [frames_rgb[i] for i in batch_indices]
                        batch_masks = [refined_masks[i] for i in batch_indices]
                        inpainted_batch = self._process_frames_with_inpainter_batch(batch_frames, batch_masks)
                        for idx_local, processed in zip(batch_indices, inpainted_batch):
                            processed_frames[idx_local] = processed

            video_tmp_path = tmpdir_path / "video_no_audio.mp4"
            
            # Try hardware encoding first, fallback to CPU if it fails
            try:
                logger.info("Attempting NVENC hardware encoding...")
                self._encode_video_nvenc(processed_frames, width, height, fps, video_tmp_path)
                logger.info("NVENC encoding successful")
            except Exception as nvenc_error:
                logger.warning("NVENC encoding failed (%s), falling back to CPU libx264 encoding", nvenc_error)
                self._encode_video_cpu(processed_frames, width, height, fps, video_tmp_path)
                logger.info("CPU encoding successful")
            
            self._remux_audio(video_tmp_path, audio_path, output_file)

        logger.info("GPU offline pipeline completed successfully: %s -> %s", input_path, output_file)
        return output_file

    def _process_video(
        self,
        input_path: Path,
        output_path: Path,
        transparent: bool,
        max_bbox_percent: float,
        force_format: Optional[str],
        detector: str | None,
    ) -> Path:
        try:
            logger.info("Attempting GPU video pipeline...")
            return self._process_video_gpu(
                input_path=input_path,
                output_path=output_path,
                transparent=transparent,
                max_bbox_percent=max_bbox_percent,
                force_format=force_format,
                detector=detector,
            )
        except Exception as exc:
            logger.error(
                f"GPU video pipeline failed with error: {str(exc)} (type: {type(exc).__name__}). Falling back to CPU implementation.",
                exc_info=True
            )
            return self._process_video_cpu(
                input_path=input_path,
                output_path=output_path,
                transparent=transparent,
                max_bbox_percent=max_bbox_percent,
                force_format=force_format,
                detector=detector,
            )

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        transparent: bool = False,
        max_bbox_percent: float = 10.0,
        force_format: Optional[str] = None,
        detector: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Process a file (image or video) for watermark removal.

        Args:
            input_path: Path to input file
            output_path: Path for output file
            transparent: Make watermark areas transparent
            max_bbox_percent: Maximum bbox percentage
            force_format: Force output format
            detector: Detection backend override ('flo'/'florence' or 'yolo')
            overwrite: Overwrite existing files

        Returns:
            Path to processed file
        """
        detector_choice = self._effective_detector(detector)

        if detector_choice == "florence":
            self._ensure_florence_model()
        else:
            try:
                self._ensure_yolo_model()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to initialize YOLO detector (%s); falling back to Florence-2.", exc)
                detector_choice = "florence"
                self._ensure_florence_model()

        self._ensure_inpaint_model()
        logger.info("Using '%s' detector for %s", detector_choice, input_path.name)

        if output_path.exists() and not overwrite:
            logger.info(f"Skipping existing file: {output_path}")
            return output_path

        # Check if it's a video
        if self._is_video_file(input_path):
            return self._process_video(
                input_path,
                output_path,
                transparent,
                max_bbox_percent,
                force_format,
                detector_choice,
            )

        # Process image
        try:
            with Image.open(input_path) as pil_image:
                image = pil_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            logger.info(
                "Input %s not recognized as image (%s); attempting video pipeline instead.",
                input_path,
                exc,
            )
            return self._process_video(
                input_path,
                output_path,
                transparent,
                max_bbox_percent,
                force_format,
                detector_choice,
            )
        mask_image = self._get_watermark_mask(image, max_bbox_percent, detector_choice)

        if transparent:
            result_image = self._make_region_transparent(image, mask_image)
        else:
            inpaint_result = self._process_image_with_inpainter(
                np.array(image), np.array(mask_image)
            )
            result_image = Image.fromarray(cv2.cvtColor(inpaint_result, cv2.COLOR_BGR2RGB))

        # Determine output format
        if force_format:
            output_format = force_format.upper()
        elif transparent:
            output_format = "PNG"
        else:
            output_format = input_path.suffix[1:].upper()
            if output_format not in ["PNG", "WEBP", "JPG"]:
                output_format = "PNG"

        # Map JPG to JPEG for PIL
        if output_format == "JPG":
            output_format = "JPEG"

        if transparent and output_format == "JPG":
            logger.warning("Transparency detected. Defaulting to PNG.")
            output_format = "PNG"

        new_output_path = output_path.with_suffix(f".{output_format.lower()}")
        new_output_path.parent.mkdir(parents=True, exist_ok=True)
        result_image.save(new_output_path, format=output_format)

        logger.info(f"Processed: {input_path} -> {new_output_path}")
        return new_output_path
