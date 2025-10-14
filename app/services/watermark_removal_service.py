"""Service for AI-powered watermark detection and removal using Florence-2 and LaMA models."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
from enum import Enum

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import tqdm


class TaskType(str, Enum):
    """Task types for Florence-2 model."""
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


class WatermarkRemovalService:
    """Service for detecting and removing watermarks from videos using AI models."""

    def __init__(self, device: str = "auto"):
        """Initialize the watermark removal service.

        Args:
            device: Device to run models on ('cuda', 'cpu', or 'auto')
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.florence_model: Optional[AutoModelForCausalLM] = None
        self.florence_processor: Optional[AutoProcessor] = None
        self.lama_model_manager: Optional[ModelManager] = None
        logger.info(f"Using device: {self.device}")

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

        if self.lama_model_manager is None:
            logger.info("Loading LaMA model...")
            self.lama_model_manager = ModelManager(name="lama", device=self.device)
            logger.info("LaMA model loaded")

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
        text_input = "watermark"
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

    def _process_image_with_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Remove watermarks from an image using LaMA inpainting.

        Args:
            image: Input image as numpy array
            mask: Binary mask as numpy array

        Returns:
            Processed image as numpy array
        """
        config = Config(
            ldm_steps=50,
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=64,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=1600,
        )

        result = self.lama_model_manager(image, mask, config)

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
                    lama_result = self._process_image_with_lama(
                        np.array(pil_image), np.array(mask_image)
                    )
                    result_image = Image.fromarray(
                        cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB)
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
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_path),  # Processed video without audio
                "-i", str(input_path),       # Original video with audio
                "-c:v", "copy",              # Copy video without re-encoding
                "-c:a", "aac",               # Encode audio to AAC
                "-map", "0:v:0",             # Use video from first input
                "-map", "1:a:0",             # Use audio from second input
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
        image = Image.open(input_path).convert("RGB")
        mask_image = self._get_watermark_mask(image, max_bbox_percent)

        if transparent:
            result_image = self._make_region_transparent(image, mask_image)
        else:
            lama_result = self._process_image_with_lama(
                np.array(image), np.array(mask_image)
            )
            result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))

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