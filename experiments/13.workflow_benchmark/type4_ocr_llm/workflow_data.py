"""OCR+LLM workflow data structures and utilities."""

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class OCRLLMWorkflowData:
    """Workflow-level data for OCR+LLM pattern (A → B).

    Stage A: OCR - Extract text from image
    Stage B: LLM - Process extracted text

    This is a simple sequential workflow without loops.
    """

    workflow_id: str
    image_data: str  # Base64-encoded image data

    # Results from each stage (populated by receivers)
    a_result: Optional[str] = None  # OCR extracted text
    b_result: Optional[str] = None  # LLM processed text

    # Timing tracking
    a_submit_time: Optional[float] = None
    a_complete_time: Optional[float] = None
    b_submit_time: Optional[float] = None
    b_complete_time: Optional[float] = None
    workflow_complete_time: Optional[float] = None

    # Additional fields for simulation mode
    a_sleep_time: Optional[float] = None  # A task sleep time (simulation)
    b_sleep_time: Optional[float] = None  # B task sleep time (simulation)

    # Additional fields for real mode
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_detail_level: str = "standard"
    max_tokens: int = 512  # Max tokens for LLM task

    # Additional fields for tracking
    strategy: str = "probabilistic"  # Scheduling strategy name
    is_warmup: bool = False  # Whether this is a warmup workflow

    def is_complete(self) -> bool:
        """Check if workflow is complete (B task done)."""
        return self.b_complete_time is not None

    def get_workflow_time(self) -> Optional[float]:
        """Get total workflow time from A submit to B complete."""
        if self.a_submit_time and self.workflow_complete_time:
            return self.workflow_complete_time - self.a_submit_time
        return None

    def get_a_latency(self) -> Optional[float]:
        """Get A task latency (submit to complete)."""
        if self.a_submit_time and self.a_complete_time:
            return self.a_complete_time - self.a_submit_time
        return None

    def get_b_latency(self) -> Optional[float]:
        """Get B task latency (submit to complete)."""
        if self.b_submit_time and self.b_complete_time:
            return self.b_complete_time - self.b_submit_time
        return None


def pre_generate_workflows(
    config,
    images: List[str],
    seed: int = 42
) -> List["OCRLLMWorkflowData"]:
    """Pre-generate all workflow data before strategy testing.

    This ensures all strategies use identical workflow data (same sleep times,
    image assignments, etc.) for fair comparison.

    Args:
        config: OCRLLMConfig instance
        images: List of base64-encoded images to use
        seed: Random seed for reproducibility

    Returns:
        List of pre-generated OCRLLMWorkflowData instances
    """
    import random

    # Set random seed for reproducibility
    random.seed(seed)

    workflows = []
    for i in range(config.num_workflows):
        workflow = OCRLLMWorkflowData(
            workflow_id=f"workflow-{i:04d}",
            image_data=images[i % len(images)],
            strategy="pending",  # Will be set per-strategy run
            ocr_languages=config.ocr_languages.split(",") if isinstance(config.ocr_languages, str) else config.ocr_languages,
            ocr_detail_level=getattr(config, 'ocr_detail_level', 'standard'),
            max_tokens=getattr(config, 'max_tokens', 512),
            is_warmup=(i < getattr(config, 'num_warmup', 0))
        )

        # Pre-generate sleep times for simulation mode
        if config.mode == "simulation":
            workflow.a_sleep_time = config.sample_sleep_time_a()
            workflow.b_sleep_time = config.sample_sleep_time_b()

        workflows.append(workflow)

    return workflows


def load_images_from_directory(directory: str, max_count: Optional[int] = None) -> List[str]:
    """Load images from a directory and encode them as base64.

    Args:
        directory: Path to directory containing images
        max_count: Maximum number of images to load (None for all)

    Returns:
        List of base64-encoded image strings

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid images found
    """
    path = Path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    images = []
    for img_path in sorted(path.iterdir()):
        if img_path.suffix.lower() in extensions:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                images.append(image_base64)

            if max_count and len(images) >= max_count:
                break

    if not images:
        raise ValueError(f"No valid images found in {directory}")

    return images


def load_images_from_json(filepath: str) -> List[str]:
    """Load base64-encoded images from a JSON file.

    The JSON file should contain either:
    - A list of base64 strings: ["base64_1", "base64_2", ...]
    - An object with 'images' key: {"images": ["base64_1", "base64_2", ...]}

    Args:
        filepath: Path to JSON file containing images

    Returns:
        List of base64-encoded image strings

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If file format is invalid
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    with open(path) as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, list):
        images = data
    elif isinstance(data, dict) and "images" in data:
        images = data["images"]
    else:
        raise ValueError(
            f"Invalid image file format. Expected list or dict with 'images' key, "
            f"got {type(data).__name__}"
        )

    if not images:
        raise ValueError("Image file is empty")

    return images


def generate_dummy_images(count: int = 100, size: tuple = (100, 100)) -> List[str]:
    """Generate dummy base64-encoded images for testing.

    Creates simple gradient images that can be used for testing
    without requiring actual image files.

    Args:
        count: Number of images to generate
        size: Image size as (width, height)

    Returns:
        List of base64-encoded PNG image strings
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        # If PIL not available, generate minimal valid PNG
        # This is a 1x1 transparent PNG
        minimal_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return [base64.b64encode(minimal_png).decode("utf-8")] * count

    images = []
    width, height = size

    for i in range(count):
        # Create a simple gradient image with varying colors
        img = Image.new("RGB", (width, height))
        pixels = img.load()

        # Create a gradient based on index
        r_base = (i * 17) % 256
        g_base = (i * 23) % 256
        b_base = (i * 31) % 256

        for x in range(width):
            for y in range(height):
                r = (r_base + x) % 256
                g = (g_base + y) % 256
                b = (b_base + x + y) % 256
                pixels[x, y] = (r, g, b)

        # Add some text-like patterns for OCR to potentially detect
        # (simple horizontal lines that might look like text)
        for y in range(10, height - 10, 20):
            for x in range(10, width - 10):
                if (x // 5) % 2 == 0:
                    pixels[x, y] = (0, 0, 0)

        # Encode as PNG
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images.append(image_base64)

    return images
