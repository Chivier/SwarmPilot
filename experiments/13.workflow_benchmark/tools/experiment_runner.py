"""
Unified experiment runner for workflow benchmarks.

This module provides a high-level interface for running Text2Video and Deep Research
experiments in both simulation and real modes.

Uses CLI arguments to invoke scripts, matching the unified interface in cli_utils.py.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger


class ExperimentRunner:
    """
    Unified runner for workflow experiments.

    Supports:
    - Text2Video workflow (simulation and real modes)
    - Deep Research workflow (simulation and real modes)
    - Automatic service management (optional)
    - Metrics collection and export

    All parameters match the unified CLI defined in common/cli_utils.py.
    """

    def __init__(self,
                 workspace_dir: Optional[Union[str, Path]] = None,
                 custom_logger = None):
        """
        Initialize experiment runner.

        Args:
            workspace_dir: Root directory for experiments (default: current directory)
            custom_logger: Optional custom logger (defaults to loguru logger)
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        self.logger = custom_logger or logger

    def _build_common_args(self,
                           num_workflows: int,
                           qps: float,
                           seed: int,
                           strategies: str,
                           warmup: float,
                           duration: int,
                           portion_stats: float = 1.0) -> List[str]:
        """Build common CLI arguments."""
        return [
            "--num-workflows", str(num_workflows),
            "--qps", str(qps),
            "--seed", str(seed),
            "--strategies", strategies,
            "--warmup", str(warmup),
            "--duration", str(duration),
            "--portion-stats", str(portion_stats),
        ]

    def _run_script(self, script_path: Path, args: List[str], description: str) -> Dict:
        """Run a script with given arguments and return results."""
        cmd = [sys.executable, str(script_path)] + args

        self.logger.info(f"Command: {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(self.workspace_dir),
            env=dict(os.environ),
        )
        elapsed = time.time() - start_time

        self.logger.info(f"Experiment completed in {elapsed:.2f}s")

        if result.returncode != 0:
            self.logger.error(f"Experiment failed with exit code {result.returncode}")
            raise RuntimeError(f"Experiment failed with exit code {result.returncode}")

        # Default metrics path (scripts may override via config)
        metrics_path = self.workspace_dir / "output" / "metrics.json"

        return {
            "success": True,
            "elapsed_time": elapsed,
            "metrics_path": str(metrics_path),
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def run_text2video_simulation(self,
                                   num_workflows: int = 10,
                                   qps: float = 2.0,
                                   seed: int = 42,
                                   strategies: str = "all",
                                   warmup: float = 0.2,
                                   duration: int = 120,
                                   max_b_loops: int = 3,
                                   portion_stats: float = 1.0) -> Dict:
        """
        Run Text2Video workflow in simulation mode.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            max_b_loops: Maximum B task iterations (default: 3)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Text2Video Simulation")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, max_b_loops={max_b_loops}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type1_text2video" / "simulation" / "test_workflow_sim.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--max-b-loops", str(max_b_loops)])

        return self._run_script(script_path, args, "Text2Video Simulation")

    def run_text2video_real(self,
                            num_workflows: int = 10,
                            qps: float = 2.0,
                            seed: int = 42,
                            strategies: str = "all",
                            warmup: float = 0.2,
                            duration: int = 120,
                            max_b_loops: int = 3,
                            portion_stats: float = 1.0) -> Dict:
        """
        Run Text2Video workflow in real cluster mode.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            max_b_loops: Maximum B task iterations (default: 3)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running Text2Video Real Cluster Mode")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, max_b_loops={max_b_loops}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type1_text2video" / "real" / "test_workflow_real.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--max-b-loops", str(max_b_loops)])

        return self._run_script(script_path, args, "Text2Video Real")

    def run_deep_research_simulation(self,
                                      num_workflows: int = 10,
                                      qps: float = 2.0,
                                      seed: int = 42,
                                      strategies: str = "all",
                                      warmup: float = 0.2,
                                      duration: int = 120,
                                      fanout: int = 4,
                                      fanout_config: Optional[str] = None,
                                      fanout_seed: Optional[int] = None,
                                      portion_stats: float = 1.0) -> Dict:
        """
        Run Deep Research workflow in simulation mode.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            fanout: Default fanout count for parallel B tasks (default: 4)
            fanout_config: Path to JSON config file for fanout distribution (optional)
            fanout_seed: Random seed for fanout distribution sampling (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        fanout_info = f"fanout={fanout}"
        if fanout_config:
            fanout_info = f"fanout_config={fanout_config}"

        self.logger.info("=" * 80)
        self.logger.info("Running Deep Research Simulation")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, {fanout_info}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type2_deep_research" / "simulation" / "test_workflow_sim.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--fanout", str(fanout)])

        if fanout_config:
            args.extend(["--fanout-config", fanout_config])
        if fanout_seed is not None:
            args.extend(["--fanout-seed", str(fanout_seed)])

        return self._run_script(script_path, args, "Deep Research Simulation")

    def run_deep_research_real(self,
                                num_workflows: int = 10,
                                qps: float = 2.0,
                                seed: int = 42,
                                strategies: str = "all",
                                warmup: float = 0.2,
                                duration: int = 120,
                                fanout: int = 4,
                                fanout_config: Optional[str] = None,
                                fanout_seed: Optional[int] = None,
                                portion_stats: float = 1.0) -> Dict:
        """
        Run Deep Research workflow in real cluster mode.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            fanout: Default fanout count for parallel B tasks (default: 4)
            fanout_config: Path to JSON config file for fanout distribution (optional)
            fanout_seed: Random seed for fanout distribution sampling (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        fanout_info = f"fanout={fanout}"
        if fanout_config:
            fanout_info = f"fanout_config={fanout_config}"

        self.logger.info("=" * 80)
        self.logger.info("Running Deep Research Real Cluster Mode")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, {fanout_info}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type2_deep_research" / "real" / "test_workflow_real.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--fanout", str(fanout)])

        if fanout_config:
            args.extend(["--fanout-config", fanout_config])
        if fanout_seed is not None:
            args.extend(["--fanout-seed", str(fanout_seed)])

        return self._run_script(script_path, args, "Deep Research Real")


if __name__ == "__main__":
    # Example usage
    from common.utils import configure_logging
    configure_logging(level="INFO")

    runner = ExperimentRunner()

    # Run Text2Video simulation
    result = runner.run_text2video_simulation(
        num_workflows=10,
        qps=2.0,
        duration=60,
        strategies="min_time,probabilistic",
    )

    print(f"\nExperiment Results:")
    print(f"Success: {result['success']}")
    print(f"Elapsed: {result['elapsed_time']:.2f}s")
    print(f"Metrics: {result['metrics_path']}")
