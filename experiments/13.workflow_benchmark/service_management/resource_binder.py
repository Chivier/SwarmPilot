"""Resource binding utilities for CPU and GPU allocation."""

import os
import logging
import subprocess
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ResourceAllocation:
    """Resource allocation for a service."""
    cpu_cores: List[int]  # List of CPU core IDs
    gpu_ids: List[int]    # List of GPU device IDs


class ResourceBinder:
    """
    Manages CPU and GPU resource allocation.

    Supports:
    - CPU core assignment
    - GPU device assignment
    - Resource validation
    - taskset command generation
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize resource binder.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Detect available resources
        self.total_cpus = self._get_cpu_count()
        self.total_gpus = self._get_gpu_count()

        self.logger.info(f"Detected {self.total_cpus} CPUs, {self.total_gpus} GPUs")

    def _get_cpu_count(self) -> int:
        """Get total number of CPU cores."""
        try:
            return os.cpu_count() or 1
        except:
            return 1

    def _get_gpu_count(self) -> int:
        """Get total number of GPU devices."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Count lines starting with "GPU"
                return len([line for line in result.stdout.split('\n') if line.startswith("GPU")])

        except:
            pass

        # Fallback: check CUDA_VISIBLE_DEVICES
        cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            return len(cuda_devices.split(','))

        return 0

    def allocate_cpus(
        self,
        num_cores: int,
        start_core: int = 0,
        exclusive: bool = False
    ) -> List[int]:
        """
        Allocate CPU cores.

        Args:
            num_cores: Number of cores to allocate
            start_core: Starting core ID
            exclusive: If True, allocate contiguous cores

        Returns:
            List of allocated core IDs

        Raises:
            ValueError: If allocation is invalid
        """
        if num_cores <= 0:
            raise ValueError("num_cores must be positive")

        if start_core < 0 or start_core >= self.total_cpus:
            raise ValueError(f"Invalid start_core: {start_core} (total: {self.total_cpus})")

        if start_core + num_cores > self.total_cpus:
            raise ValueError(
                f"Cannot allocate {num_cores} cores starting at {start_core} "
                f"(total: {self.total_cpus})"
            )

        cores = list(range(start_core, start_core + num_cores))

        self.logger.info(f"Allocated CPU cores: {cores}")
        return cores

    def allocate_gpus(
        self,
        num_gpus: int,
        start_gpu: int = 0
    ) -> List[int]:
        """
        Allocate GPU devices.

        Args:
            num_gpus: Number of GPUs to allocate
            start_gpu: Starting GPU ID

        Returns:
            List of allocated GPU IDs

        Raises:
            ValueError: If allocation is invalid
        """
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")

        if self.total_gpus == 0:
            raise ValueError("No GPUs available")

        if start_gpu < 0 or start_gpu >= self.total_gpus:
            raise ValueError(f"Invalid start_gpu: {start_gpu} (total: {self.total_gpus})")

        if start_gpu + num_gpus > self.total_gpus:
            raise ValueError(
                f"Cannot allocate {num_gpus} GPUs starting at {start_gpu} "
                f"(total: {self.total_gpus})"
            )

        gpus = list(range(start_gpu, start_gpu + num_gpus))

        self.logger.info(f"Allocated GPUs: {gpus}")
        return gpus

    def cpu_list_to_taskset_format(self, cores: List[int]) -> str:
        """
        Convert CPU core list to taskset format.

        Args:
            cores: List of core IDs

        Returns:
            Taskset format string (e.g., "0-3" or "0,2,4")

        Examples:
            [0, 1, 2, 3] -> "0-3"
            [0, 2, 4] -> "0,2,4"
            [0, 1, 3, 4] -> "0-1,3-4"
        """
        if not cores:
            return ""

        cores = sorted(cores)

        # Try to find contiguous ranges
        ranges = []
        start = cores[0]
        end = cores[0]

        for i in range(1, len(cores)):
            if cores[i] == end + 1:
                # Extend current range
                end = cores[i]
            else:
                # Start new range
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")

                start = cores[i]
                end = cores[i]

        # Add last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ",".join(ranges)

    def gpu_list_to_env_format(self, gpus: List[int]) -> str:
        """
        Convert GPU list to CUDA_VISIBLE_DEVICES format.

        Args:
            gpus: List of GPU IDs

        Returns:
            Comma-separated GPU IDs (e.g., "0,1,2")
        """
        return ",".join(str(g) for g in gpus)

    def create_allocation(
        self,
        num_cpus: int,
        num_gpus: int = 0,
        cpu_offset: int = 0,
        gpu_offset: int = 0
    ) -> ResourceAllocation:
        """
        Create a resource allocation.

        Args:
            num_cpus: Number of CPU cores
            num_gpus: Number of GPUs
            cpu_offset: CPU starting offset
            gpu_offset: GPU starting offset

        Returns:
            ResourceAllocation object
        """
        cpu_cores = self.allocate_cpus(num_cpus, start_core=cpu_offset)
        gpu_ids = self.allocate_gpus(num_gpus, start_gpu=gpu_offset) if num_gpus > 0 else []

        return ResourceAllocation(cpu_cores=cpu_cores, gpu_ids=gpu_ids)

    def validate_allocation(self, allocation: ResourceAllocation) -> bool:
        """
        Validate a resource allocation.

        Args:
            allocation: ResourceAllocation to validate

        Returns:
            True if valid

        Raises:
            ValueError: If allocation is invalid
        """
        # Check CPUs
        for core in allocation.cpu_cores:
            if core < 0 or core >= self.total_cpus:
                raise ValueError(f"Invalid CPU core: {core} (total: {self.total_cpus})")

        # Check GPUs
        for gpu in allocation.gpu_ids:
            if gpu < 0 or gpu >= self.total_gpus:
                raise ValueError(f"Invalid GPU: {gpu} (total: {self.total_gpus})")

        return True
