"""Tests for hardware performance information utilities."""

import pytest

from swarmpilot.predictor.utils.hardware_perf_info import (
    NVIDIA_TESLA_SPECS,
    get_gpu_spec,
)


class TestGetGpuSpec:
    """Tests for get_gpu_spec function."""

    def test_get_gpu_spec_valid_gpu(self):
        """Should return specs for a valid GPU model."""
        spec = get_gpu_spec("H100")

        assert "cuda_cores" in spec
        assert "tensor_cores" in spec
        assert "fp32_tflops" in spec
        assert "memory_gb" in spec
        assert "memory_bandwidth_gb_s" in spec
        assert spec["cuda_cores"] == 14592
        assert spec["memory_gb"] == 80

    def test_get_gpu_spec_all_gpus(self):
        """Should return valid specs for all available GPUs."""
        for gpu_name in NVIDIA_TESLA_SPECS:
            spec = get_gpu_spec(gpu_name)
            assert "cuda_cores" in spec
            assert "fp32_tflops" in spec
            assert "memory_gb" in spec
            assert spec["cuda_cores"] > 0
            assert spec["fp32_tflops"] > 0
            assert spec["memory_gb"] > 0

    def test_get_gpu_spec_invalid_gpu(self):
        """Should raise KeyError for invalid GPU model."""
        with pytest.raises(KeyError, match="not found"):
            get_gpu_spec("INVALID_GPU")

    def test_get_gpu_spec_v100(self):
        """Should return correct specs for V100."""
        spec = get_gpu_spec("V100")

        assert spec["cuda_cores"] == 5120
        assert spec["tensor_cores"] == 640
        assert spec["fp32_tflops"] == 15.7
        assert spec["memory_gb"] == 16

    def test_get_gpu_spec_a100(self):
        """Should return correct specs for A100."""
        spec = get_gpu_spec("A100")

        assert spec["cuda_cores"] == 6912
        assert spec["tensor_cores"] == 432
        assert spec["memory_gb"] == 40

    def test_get_gpu_spec_t4(self):
        """Should return correct specs for T4."""
        spec = get_gpu_spec("T4")

        assert spec["cuda_cores"] == 2560
        assert spec["memory_gb"] == 16


class TestNvidiaTeslaSpecs:
    """Tests for the NVIDIA_TESLA_SPECS constant."""

    def test_all_gpus_have_required_fields(self):
        """All GPUs should have the required specification fields."""
        required_fields = [
            "cuda_cores",
            "tensor_cores",
            "fp32_tflops",
            "memory_gb",
            "memory_bandwidth_gb_s",
        ]

        for gpu_name, spec in NVIDIA_TESLA_SPECS.items():
            for field in required_fields:
                assert field in spec, f"{gpu_name} missing {field}"

    def test_all_values_are_positive(self):
        """All specification values should be positive numbers."""
        for gpu_name, spec in NVIDIA_TESLA_SPECS.items():
            for key, value in spec.items():
                assert value > 0, (
                    f"{gpu_name}.{key} should be positive, got {value}"
                )

    def test_h100_variants_exist(self):
        """H100 should have multiple variants."""
        assert "H100" in NVIDIA_TESLA_SPECS
        assert "H100-PCIe" in NVIDIA_TESLA_SPECS
        assert "H100-94GB" in NVIDIA_TESLA_SPECS

    def test_a100_variants_exist(self):
        """A100 should have multiple variants."""
        assert "A100" in NVIDIA_TESLA_SPECS
        assert "A100-80GB" in NVIDIA_TESLA_SPECS
