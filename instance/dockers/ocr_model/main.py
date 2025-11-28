"""
OCR Model Container - Text Extraction from Images using EasyOCR

A FastAPI service that performs Optical Character Recognition using EasyOCR.
This service extracts text from images and provides structured output for
downstream LLM processing.

Input:
- image_data: Base64-encoded image data (PNG, JPEG, etc.)
- languages: List of language codes (default: ["en"])
- confidence_threshold: Minimum confidence threshold (0.0-1.0)

Output:
- extracted_text: Concatenated text from all detected blocks
- text_blocks: List of detected text with confidence scores
- block_count: Number of text blocks detected
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from argparse import ArgumentParser

import numpy as np
from PIL import Image

# EasyOCR import
try:
    import easyocr
except ImportError:
    easyocr = None


parser = ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = os.getenv("MODEL_ID", "ocr_model")
INSTANCE_ID = os.getenv("INSTANCE_ID", "unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_LANGUAGES = os.getenv("OCR_DEFAULT_LANGUAGES", "en").split(",")
GPU_ENABLED = os.getenv("OCR_GPU_ENABLED", "false").lower() == "true"

# Model state
reader: Optional[Any] = None
loaded_languages: List[str] = []


# =============================================================================
# Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    global reader, loaded_languages

    # Startup
    print(f"OCR Model Container starting...")
    print(f"Model ID: {MODEL_ID}")
    print(f"Instance ID: {INSTANCE_ID}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"GPU Enabled: {GPU_ENABLED}")

    if easyocr is None:
        print("ERROR: EasyOCR is not installed. Please install it with: pip install easyocr")
        yield
        return

    # Initialize EasyOCR with default languages
    print(f"Loading EasyOCR with languages: {DEFAULT_LANGUAGES}")
    try:
        reader = easyocr.Reader(
            DEFAULT_LANGUAGES,
            gpu=GPU_ENABLED,
            verbose=False
        )
        loaded_languages = DEFAULT_LANGUAGES.copy()
        print("EasyOCR loaded successfully!")
    except Exception as e:
        print(f"Failed to load EasyOCR: {e}")
        reader = None

    print("OCR Model Container is running")

    yield

    # Shutdown
    print("OCR Model Container shutting down...")
    reader = None
    loaded_languages = []
    print("Shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=f"OCR Model Container - {MODEL_ID}",
    description="OCR service using EasyOCR for text extraction from images",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Request/Response Models
# =============================================================================

class InferenceRequest(BaseModel):
    """Request schema for OCR inference"""

    image_data: str = Field(
        ...,
        description="Base64-encoded image data (PNG, JPEG, etc.)",
        min_length=1
    )

    languages: List[str] = Field(
        default=["en"],
        description="List of language codes for OCR (e.g., ['en'], ['ch_sim', 'en'])",
        min_length=1,
        max_length=10
    )

    detail_level: str = Field(
        default="standard",
        description="Output detail: 'minimal' (text only), 'standard' (text + confidence), 'full' (text + confidence + bbox)"
    )

    confidence_threshold: float = Field(
        default=0.0,
        description="Minimum confidence threshold (0.0-1.0)",
        ge=0.0,
        le=1.0
    )

    paragraph_mode: bool = Field(
        default=True,
        description="Group text into paragraphs"
    )


class InferenceResponse(BaseModel):
    """Response schema for inference"""
    success: bool
    result: Dict[str, Any]
    execution_time: float
    start_timestamp: int


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool


# =============================================================================
# Helper Functions
# =============================================================================

def decode_image(base64_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    # Handle data URL format (e.g., "data:image/png;base64,...")
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (EasyOCR expects RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return np.array(image)


def format_ocr_results(
    raw_results: List,
    detail_level: str,
    confidence_threshold: float,
    paragraph_mode: bool
) -> Dict[str, Any]:
    """Format EasyOCR results based on detail level."""

    # Filter by confidence threshold
    filtered_results = [
        r for r in raw_results
        if r[2] >= confidence_threshold
    ]

    # Build text blocks
    text_blocks = []
    texts = []
    confidences = []

    for bbox, text, confidence in filtered_results:
        texts.append(text)
        confidences.append(confidence)

        block = {"text": text, "confidence": round(confidence, 4)}
        if detail_level == "full":
            # Convert bbox to list of lists
            block["bbox"] = [[int(p[0]), int(p[1])] for p in bbox]

        if detail_level != "minimal":
            text_blocks.append(block)

    # Concatenate text
    separator = "\n\n" if paragraph_mode else "\n"
    extracted_text = separator.join(texts)

    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    result = {
        "extracted_text": extracted_text,
        "block_count": len(filtered_results),
        "average_confidence": round(avg_confidence, 4),
    }

    if detail_level != "minimal":
        result["text_blocks"] = text_blocks

    return result


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    """
    Execute OCR inference on the provided image.

    Args:
        request: InferenceRequest with image_data and OCR parameters

    Returns:
        InferenceResponse with extracted text and execution details
    """
    if reader is None:
        raise HTTPException(
            status_code=503,
            detail="EasyOCR model not loaded"
        )

    try:
        # Record start time
        start_time = time.time()
        start_timestamp = int(start_time)

        # Decode image
        try:
            image_array = decode_image(request.image_data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )

        # Check if we need to reload reader with different languages
        # Note: In production, consider caching readers for common language sets
        current_reader = reader
        languages_used = loaded_languages

        if set(request.languages) != set(loaded_languages):
            # For simplicity, use loaded languages with a warning
            # In production, could maintain a cache of readers
            print(f"Warning: Requested languages {request.languages} differ from loaded {loaded_languages}")
            languages_used = loaded_languages

        # Run OCR
        raw_results = current_reader.readtext(
            image_array,
            paragraph=request.paragraph_mode
        )

        # Format results
        formatted = format_ocr_results(
            raw_results,
            request.detail_level,
            request.confidence_threshold,
            request.paragraph_mode
        )

        # Calculate execution time
        execution_time = time.time() - start_time

        # Build final result
        result = {
            **formatted,
            "languages_used": languages_used,
            "model_id": MODEL_ID,
            "instance_id": INSTANCE_ID,
            "start_time": start_time
        }

        return InferenceResponse(
            success=True,
            result=result,
            execution_time=execution_time,
            start_timestamp=start_timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR inference failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse indicating model status
    """
    if reader is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level=LOG_LEVEL.lower()
    )
