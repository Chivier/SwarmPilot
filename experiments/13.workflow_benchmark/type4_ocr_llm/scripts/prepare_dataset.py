#!/usr/bin/env python3
"""
OmniDocBench Dataset Preparation Script

Downloads document images from opendatalab/OmniDocBench HuggingFace dataset
for use with the OCR+LLM workflow.

Dataset info:
- Source: https://huggingface.co/datasets/opendatalab/OmniDocBench
- Size: ~1,355 images, ~1.25GB
- Format: PNG images of document pages

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --output-dir ./custom_data
"""

import argparse
import shutil
from pathlib import Path


def check_disk_space(path: Path, required_gb: float = 2.0) -> None:
    """Check if enough disk space is available.

    Args:
        path: Target directory path
        required_gb: Required space in GB

    Raises:
        RuntimeError: If insufficient disk space
    """
    parent = path
    while not parent.exists():
        parent = parent.parent

    usage = shutil.disk_usage(parent)
    available_gb = usage.free / (1024**3)

    if available_gb < required_gb:
        raise RuntimeError(
            f"Insufficient disk space. Need {required_gb:.1f}GB, have {available_gb:.1f}GB"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniDocBench images for OCR+LLM workflow"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/omnidocbench",
        help="Output directory for downloaded data (default: data/omnidocbench)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: uses HF_HOME)",
    )
    args = parser.parse_args()

    # Resolve output path relative to script location
    script_dir = Path(__file__).parent.parent
    output_path = script_dir / args.output_dir

    print("=" * 60)
    print("OmniDocBench Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print()

    # Step 1: Check disk space
    print("[1/4] Checking disk space...")
    check_disk_space(output_path, required_gb=2.0)
    print("      Disk space OK (need ~2GB)")

    # Step 2: Import huggingface_hub (deferred to provide better error message)
    print("[2/4] Loading HuggingFace Hub...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return 1

    # Step 3: Download images from HuggingFace
    print("[3/4] Downloading OmniDocBench images...")
    print("      Repository: opendatalab/OmniDocBench")
    print("      This may take a few minutes (~1.25GB)...")
    print()

    local_path = snapshot_download(
        repo_id="opendatalab/OmniDocBench",
        repo_type="dataset",
        allow_patterns=["images/*"],  # Only download images directory
        local_dir=str(output_path),
        cache_dir=args.cache_dir,
    )

    # Step 4: Verify download
    print()
    print("[4/4] Verifying download...")
    images_dir = Path(local_path) / "images"

    if not images_dir.exists():
        print(f"ERROR: Images directory not found at {images_dir}")
        return 1

    # Count images (PNG, JPG, etc.)
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    image_files = [
        f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    print(f"      Found {len(image_files)} images in {images_dir}")

    # Print summary
    print()
    print("=" * 60)
    print("Dataset prepared successfully!")
    print("=" * 60)
    print()
    print(f"Images location: {images_dir}")
    print()
    print("To use with the OCR+LLM workflow:")
    print()
    print("  # From experiments/13.workflow_benchmark/")
    print(f"  python tools/cli.py run-ocr-llm-real \\")
    print(f"      --image-dir {images_dir.relative_to(script_dir.parent)} \\")
    print(f"      --num-workflows 100 \\")
    print(f"      --qps 2.0 \\")
    print(f"      --strategies probabilistic")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
