"""Hardware performance information for Nvidia Tesla series GPUs.

Contains CUDA core counts, theoretical TFLOPS, and memory specifications.
"""

from __future__ import annotations

# Nvidia Tesla Series GPU Specifications (V100 to H100)
NVIDIA_TESLA_SPECS = {
    "V100": {
        "cuda_cores": 5120,
        "tensor_cores": 640,
        "fp32_tflops": 15.7,
        "fp16_tflops": 31.4,
        "tensor_tflops": 125.0,  # FP16 with Tensor Cores
        "memory_gb": 16,  # Standard variant, also available in 32GB
        "memory_bandwidth_gb_s": 900,
    },
    "V100-32GB": {
        "cuda_cores": 5120,
        "tensor_cores": 640,
        "fp32_tflops": 15.7,
        "fp16_tflops": 31.4,
        "tensor_tflops": 125.0,
        "memory_gb": 32,
        "memory_bandwidth_gb_s": 900,
    },
    "T4": {
        "cuda_cores": 2560,
        "tensor_cores": 320,
        "fp32_tflops": 8.1,
        "fp16_tflops": 65.0,
        "tensor_tflops": 130.0,  # INT8
        "memory_gb": 16,
        "memory_bandwidth_gb_s": 300,
    },
    "A100": {
        "cuda_cores": 6912,
        "tensor_cores": 432,
        "fp32_tflops": 19.5,
        "fp16_tflops": 78.0,
        "tensor_tflops": 312.0,  # FP16 with Tensor Cores
        "memory_gb": 40,  # Standard variant, also available in 80GB
        "memory_bandwidth_gb_s": 1555,
    },
    "A100-80GB": {
        "cuda_cores": 6912,
        "tensor_cores": 432,
        "fp32_tflops": 19.5,
        "fp16_tflops": 78.0,
        "tensor_tflops": 312.0,
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 2039,
    },
    "A10": {
        "cuda_cores": 9216,
        "tensor_cores": 288,
        "fp32_tflops": 31.2,
        "fp16_tflops": 125.0,
        "tensor_tflops": 250.0,  # FP16 with Tensor Cores
        "memory_gb": 24,
        "memory_bandwidth_gb_s": 600,
    },
    "A30": {
        "cuda_cores": 3584,
        "tensor_cores": 224,
        "fp32_tflops": 10.3,
        "fp16_tflops": 82.0,
        "tensor_tflops": 165.0,
        "memory_gb": 24,
        "memory_bandwidth_gb_s": 933,
    },
    "A40": {
        "cuda_cores": 10752,
        "tensor_cores": 336,
        "fp32_tflops": 37.4,
        "fp16_tflops": 74.8,
        "tensor_tflops": 150.0,
        "memory_gb": 48,
        "memory_bandwidth_gb_s": 696,
    },
    "H100": {
        "cuda_cores": 14592,  # SM count: 114, CUDA cores per SM: 128
        "tensor_cores": 456,
        "fp32_tflops": 51.0,
        "fp16_tflops": 204.0,
        "tensor_tflops": 989.0,  # FP16 with Tensor Cores
        "fp8_tensor_tflops": 1979.0,  # FP8 with Tensor Cores
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 3350,  # HBM3
    },
    "H100-94GB": {
        "cuda_cores": 14592,
        "tensor_cores": 456,
        "fp32_tflops": 51.0,
        "fp16_tflops": 204.0,
        "tensor_tflops": 989.0,
        "fp8_tensor_tflops": 1979.0,
        "memory_gb": 94,  # HBM3e variant
        "memory_bandwidth_gb_s": 3350,
    },
    "H100-PCIe": {
        "cuda_cores": 14592,
        "tensor_cores": 456,
        "fp32_tflops": 48.0,
        "fp16_tflops": 192.0,
        "tensor_tflops": 756.0,
        "fp8_tensor_tflops": 1513.0,
        "memory_gb": 80,
        "memory_bandwidth_gb_s": 2000,  # PCIe variant has lower bandwidth
    },
    "H20": {
        "cuda_cores": 17920,  # Hopper architecture, similar to H100
        "tensor_cores": 560,
        "fp32_tflops": 63.0,
        "fp16_tflops": 252.0,
        "tensor_tflops": 1230.0,  # FP16 with Tensor Cores
        "fp8_tensor_tflops": 2460.0,  # FP8 with Tensor Cores
        "memory_gb": 96,  # HBM3e
        "memory_bandwidth_gb_s": 4000,  # HBM3e bandwidth
    },
}


def get_gpu_spec(gpu_name: str) -> dict[str, int | float]:
    """Get specifications for a specific GPU model.

    Args:
        gpu_name: Name of the GPU (e.g., V100, A100, H100).

    Returns:
        Dictionary containing GPU specifications.

    Raises:
        KeyError: If GPU model is not found.
    """
    if gpu_name not in NVIDIA_TESLA_SPECS:
        available = list(NVIDIA_TESLA_SPECS.keys())
        raise KeyError(f"GPU model '{gpu_name}' not found. Available: {available}")
    return NVIDIA_TESLA_SPECS[gpu_name]


