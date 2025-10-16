"""Service for AI-powered watermark detection and removal using Florence-2 and configurable inpainting models."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from enum import Enum

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
        "sora logo",
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
    WATERMARK_MIN_SCORE = 0.18

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
        if score is not None and score < self.WATERMARK_MIN_SCORE:
            return False

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
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
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

        detection_interval = max(1, int(round(fps)) or 1)
        combined_mask_array: np.ndarray | None = None
        mask_image_cached: Image.Image | None = None

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
                    mask_image_cached = self._get_watermark_mask(pil_image, max_bbox_percent, detector)
                    combined_mask_array = np.array(mask_image_cached, dtype=np.uint8)
                elif combined_mask_array is None:
                    combined_mask_array = np.zeros((height, width), dtype=np.uint8)
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
