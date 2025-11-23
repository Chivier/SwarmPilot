"""
Unified experiment runner for workflow benchmarks.

This module provides a high-level interface for running Text2Video and Deep Research
experiments in both simulation and real modes.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging


class ExperimentRunner:
    """
    Unified runner for workflow experiments.

    Supports:
    - Text2Video workflow (simulation and real modes)
    - Deep Research workflow (simulation and real modes)
    - Automatic service management (optional)
    - Metrics collection and export
    """

    def __init__(self,
                 workspace_dir: Optional[Union[str, Path]] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize experiment runner.

        Args:
            workspace_dir: Root directory for experiments (default: current directory)
            logger: Optional logger instance
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.logger = logger or logging.getLogger(__name__)

    def run_text2video_simulation(self,
                                   qps: float = 2.0,
                                   duration: int = 300,
                                   num_workflows: int = 600,
                                   max_b_loops: int = 4,
                                   output_dir: str = "output",
                                   **kwargs) -> Dict:
        """
        Run Text2Video workflow in simulation mode.

        Args:
            qps: Query per second rate
            duration: Experiment duration in seconds
            num_workflows: Total number of workflows to execute
            max_b_loops: Maximum B task iterations per workflow
            output_dir: Output directory for metrics
            **kwargs: Additional configuration parameters

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Text2Video Simulation")
        self.logger.info("=" * 80)

        # Prepare environment variables
        env = {
            "MODE": "simulation",
            "QPS": str(qps),
            "DURATION": str(duration),
            "NUM_WORKFLOWS": str(num_workflows),
            "MAX_B_LOOPS": str(max_b_loops),
            "OUTPUT_DIR": output_dir,
            **kwargs
        }

        # Build command
        script_path = self.workspace_dir / "type1_text2video" / "simulation" / "test_workflow_sim.py"
        cmd = [sys.executable, str(script_path)]

        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"Config: QPS={qps}, Duration={duration}s, Workflows={num_workflows}")

        # Run experiment
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(self.workspace_dir),
            env={**dict(os.environ), **env},
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time

        self.logger.info(f"Experiment completed in {elapsed:.2f}s")

        if result.returncode != 0:
            self.logger.error(f"Experiment failed: {result.stderr}")
            raise RuntimeError(f"Experiment failed with exit code {result.returncode}")

        # Parse output for metrics path
        metrics_path = Path(output_dir) / "metrics.json"

        return {
            "success": True,
            "elapsed_time": elapsed,
            "metrics_path": str(metrics_path),
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def run_text2video_real(self,
                            qps: float = 2.0,
                            duration: int = 300,
                            num_workflows: int = 100,
                            max_b_loops: int = 4,
                            output_dir: str = "output",
                            **kwargs) -> Dict:
        """
        Run Text2Video workflow in real cluster mode.

        Args:
            qps: Query per second rate
            duration: Experiment duration in seconds
            num_workflows: Total number of workflows to execute
            max_b_loops: Maximum B task iterations per workflow
            output_dir: Output directory for metrics
            **kwargs: Additional configuration (scheduler URLs, model IDs, etc.)

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Text2Video Real Cluster Mode")
        self.logger.info("=" * 80)

        # Prepare environment variables
        env = {
            "MODE": "real",
            "QPS": str(qps),
            "DURATION": str(duration),
            "NUM_WORKFLOWS": str(num_workflows),
            "MAX_B_LOOPS": str(max_b_loops),
            "OUTPUT_DIR": output_dir,
            **kwargs
        }

        # Build command
        script_path = self.workspace_dir / "type1_text2video" / "real" / "test_workflow_real.py"
        cmd = [sys.executable, str(script_path)]

        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"Config: QPS={qps}, Duration={duration}s, Workflows={num_workflows}")

        # Run experiment
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(self.workspace_dir),
            env={**dict(os.environ), **env},
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time

        self.logger.info(f"Experiment completed in {elapsed:.2f}s")

        if result.returncode != 0:
            self.logger.error(f"Experiment failed: {result.stderr}")
            raise RuntimeError(f"Experiment failed with exit code {result.returncode}")

        metrics_path = Path(output_dir) / "metrics.json"

        return {
            "success": True,
            "elapsed_time": elapsed,
            "metrics_path": str(metrics_path),
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def run_deep_research_simulation(self,
                                      qps: float = 1.0,
                                      duration: int = 600,
                                      num_workflows: int = 600,
                                      fanout_count: int = 3,
                                      output_dir: str = "output",
                                      **kwargs) -> Dict:
        """
        Run Deep Research workflow in simulation mode.

        Args:
            qps: Query per second rate
            duration: Experiment duration in seconds
            num_workflows: Total number of workflows to execute
            fanout_count: Number of B1/B2 tasks per workflow
            output_dir: Output directory for metrics
            **kwargs: Additional configuration parameters

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Deep Research Simulation")
        self.logger.info("=" * 80)

        # Prepare environment variables
        env = {
            "MODE": "simulation",
            "QPS": str(qps),
            "DURATION": str(duration),
            "NUM_WORKFLOWS": str(num_workflows),
            "FANOUT_COUNT": str(fanout_count),
            "OUTPUT_DIR": output_dir,
            **kwargs
        }

        # Build command
        script_path = self.workspace_dir / "type2_deep_research" / "simulation" / "test_workflow_sim.py"
        cmd = [sys.executable, str(script_path)]

        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"Config: QPS={qps}, Duration={duration}s, Workflows={num_workflows}, Fanout={fanout_count}")

        # Run experiment
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(self.workspace_dir),
            env={**dict(os.environ), **env},
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time

        self.logger.info(f"Experiment completed in {elapsed:.2f}s")

        if result.returncode != 0:
            self.logger.error(f"Experiment failed: {result.stderr}")
            raise RuntimeError(f"Experiment failed with exit code {result.returncode}")

        metrics_path = Path(output_dir) / "metrics.json"

        return {
            "success": True,
            "elapsed_time": elapsed,
            "metrics_path": str(metrics_path),
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def run_deep_research_real(self,
                                qps: float = 1.0,
                                duration: int = 600,
                                num_workflows: int = 100,
                                fanout_count: int = 3,
                                output_dir: str = "output",
                                **kwargs) -> Dict:
        """
        Run Deep Research workflow in real cluster mode.

        Args:
            qps: Query per second rate
            duration: Experiment duration in seconds
            num_workflows: Total number of workflows to execute
            fanout_count: Number of B1/B2 tasks per workflow
            output_dir: Output directory for metrics
            **kwargs: Additional configuration (scheduler URLs, model IDs, etc.)

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Deep Research Real Cluster Mode")
        self.logger.info("=" * 80)

        # Prepare environment variables
        env = {
            "MODE": "real",
            "QPS": str(qps),
            "DURATION": str(duration),
            "NUM_WORKFLOWS": str(num_workflows),
            "FANOUT_COUNT": str(fanout_count),
            "OUTPUT_DIR": output_dir,
            **kwargs
        }

        # Build command
        script_path = self.workspace_dir / "type2_deep_research" / "real" / "test_workflow_real.py"
        cmd = [sys.executable, str(script_path)]

        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"Config: QPS={qps}, Duration={duration}s, Workflows={num_workflows}, Fanout={fanout_count}")

        # Run experiment
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(self.workspace_dir),
            env={**dict(os.environ), **env},
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time

        self.logger.info(f"Experiment completed in {elapsed:.2f}s")

        if result.returncode != 0:
            self.logger.error(f"Experiment failed: {result.stderr}")
            raise RuntimeError(f"Experiment failed with exit code {result.returncode}")

        metrics_path = Path(output_dir) / "metrics.json"

        return {
            "success": True,
            "elapsed_time": elapsed,
            "metrics_path": str(metrics_path),
            "stdout": result.stdout,
            "stderr": result.stderr
        }


import os


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    runner = ExperimentRunner()

    # Run Text2Video simulation
    result = runner.run_text2video_simulation(
        qps=2.0,
        duration=60,
        num_workflows=120
    )

    print(f"\nExperiment Results:")
    print(f"Success: {result['success']}")
    print(f"Elapsed: {result['elapsed_time']:.2f}s")
    print(f"Metrics: {result['metrics_path']}")
