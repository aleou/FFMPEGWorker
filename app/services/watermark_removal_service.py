"""Service for AI-powered watermark detection and removal using Florence-2 and configurable inpainting models."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from enum import Enum

import cv2
import numpy as np
import torch
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


class TaskType(str, Enum):
    """Task types for Florence-2 model."""
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


class WatermarkRemovalService:
    """Service for detecting and removing watermarks from videos using AI models."""

    SUPPORTED_INPAINT_MODELS = ("lama", "zits", "cv2")

    def __init__(
        self,
        device: str = "auto",
        preferred_models: Sequence[str] | str | None = None,
    ):
        """Initialize the watermark removal service.

        Args:
            device: Device to run models on ('cuda', 'cpu', or 'auto')
            preferred_models: Preferred inpainting model(s) in order of priority.
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._cuda_capability = self._detect_cuda_capability()
        self._inpaint_candidates = self._resolve_inpaint_candidates(preferred_models)
        self._remaining_inpaint_candidates = list(self._inpaint_candidates)

        self.florence_model: Optional[AutoModelForCausalLM] = None
        self.florence_processor: Optional[AutoProcessor] = None
        self.inpaint_model_manager: Optional[ModelManager] = None
        self._inpaint_device: torch.device | None = None
        self._active_inpaint_model: str | None = None

        logger.info(
            f"Using device: {self.device} (inpaint preferences: {', '.join(self._inpaint_candidates)})"
        )

    @staticmethod
    def _coerce_device(device: str | torch.device) -> torch.device:
        return device if isinstance(device, torch.device) else torch.device(device)

    def _detect_cuda_capability(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        try:
            major, minor = torch.cuda.get_device_capability()
            return float(f"{major}.{minor}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Unable to read CUDA capability ({exc}); assuming no CUDA acceleration.")
            return None

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

    def _load_models(self):
        """Load the AI models if not already loaded."""
        if self.florence_model is None:
            logger.info("Loading Florence-2 model...")
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large", trust_remote_code=True
            ).to(self.device).eval()
            self.florence_processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large", trust_remote_code=True
            )
            logger.info("Florence-2 model loaded")

        self._ensure_inpaint_model()

    def _identify_watermark(self, image: Image.Image, text_input: str) -> dict:
        """Detect watermarks in an image using Florence-2.

        Args:
            image: PIL Image to analyze
            text_input: Text prompt for detection

        Returns:
            Parsed detection results
        """
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

    def _get_watermark_mask(self, image: Image.Image, max_bbox_percent: float) -> Image.Image:
        """Generate a mask for watermark regions.

        Args:
            image: PIL Image to process
            max_bbox_percent: Maximum percentage of image area a bbox can cover

        Returns:
            Binary mask image
        """
        text_input = "watermark logo sora"
        parsed_answer = self._identify_watermark(image, text_input)

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)

        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            image_area = image.width * image.height
            for bbox in parsed_answer[detection_key]["bboxes"]:
                x1, y1, x2, y2 = map(int, bbox)
                bbox_area = (x2 - x1) * (y2 - y1)
                if (bbox_area / image_area) * 100 <= max_bbox_percent:
                    draw.rectangle([x1, y1, x2, y2], fill=255)
                else:
                    logger.warning(
                        f"Skipping large bounding box: {bbox} covering "
                        f"{bbox_area / image_area:.2%} of the image"
                    )

        return mask

    def _process_image_with_inpainter(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove watermarks from an image using the configured inpainting model.

        Args:
            image: Input image as numpy array
            mask: Binary mask as numpy array

        Returns:
            Processed image as numpy array
        """
        # Older IOPaint builds rely on Pydantic v1 style validators and can break
        # when instantiating via the regular constructor under Pydantic v2.
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
            config = Config.model_construct(**config_defaults)
        else:  # pragma: no cover - fallback for Pydantic v1
            config = Config.construct(**config_defaults)  # type: ignore[attr-defined]

        attempts: set[str] = set()
        while True:
            self._ensure_inpaint_model()
            assert self.inpaint_model_manager is not None
            model_name = self._active_inpaint_model or "unknown"
            try:
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

    def _process_video(
        self,
        input_path: Path,
        output_path: Path,
        transparent: bool,
        max_bbox_percent: float,
        force_format: Optional[str]
    ) -> Path:
        """Process a video file for watermark removal.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            transparent: Whether to make watermarks transparent
            max_bbox_percent: Maximum bbox percentage
            force_format: Forced output format

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

                # Get watermark mask
                mask_image = self._get_watermark_mask(pil_image, max_bbox_percent)

                # Process frame
                if transparent:
                    result_image = self._make_region_transparent(pil_image, mask_image)
                    # Convert RGBA to RGB by filling with white
                    background = Image.new("RGB", result_image.size, (255, 255, 255))
                    background.paste(result_image, mask=result_image.split()[3])
                    result_image = background
                else:
                    inpaint_result = self._process_image_with_inpainter(
                        np.array(pil_image), np.array(mask_image)
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

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        transparent: bool = False,
        max_bbox_percent: float = 10.0,
        force_format: Optional[str] = None,
        overwrite: bool = False
    ) -> Path:
        """Process a file (image or video) for watermark removal.

        Args:
            input_path: Path to input file
            output_path: Path for output file
            transparent: Make watermark areas transparent
            max_bbox_percent: Maximum bbox percentage
            force_format: Force output format
            overwrite: Overwrite existing files

        Returns:
            Path to processed file
        """
        # Load models if needed
        self._load_models()

        if output_path.exists() and not overwrite:
            logger.info(f"Skipping existing file: {output_path}")
            return output_path

        # Check if it's a video
        if self._is_video_file(input_path):
            return self._process_video(input_path, output_path, transparent, max_bbox_percent, force_format)

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
            return self._process_video(input_path, output_path, transparent, max_bbox_percent, force_format)
        mask_image = self._get_watermark_mask(image, max_bbox_percent)

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
