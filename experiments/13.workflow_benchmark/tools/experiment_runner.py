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
                                   frame_count: int = 16,
                                   frame_count_config: Optional[str] = None,
                                   frame_count_seed: Optional[int] = None,
                                   max_b_loops_config: Optional[str] = None,
                                   max_b_loops_seed: Optional[int] = None,
                                   portion_stats: float = 1.0,
                                   max_sleep_time: float = 600.0) -> Dict:
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
            frame_count: Frame count for video generation (default: 16)
            frame_count_config: Path to JSON config for frame_count distribution (optional)
            frame_count_seed: Random seed for frame_count distribution (optional)
            max_b_loops_config: Path to JSON config for max_b_loops distribution (optional)
            max_b_loops_seed: Random seed for max_b_loops distribution (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)
            max_sleep_time: Maximum sleep time in seconds for scaling (default: 600.0)

        Returns:
            Dict with experiment results and metrics path
        """
        # Build distribution info string
        frame_info = f"frame_count={frame_count}"
        if frame_count_config:
            frame_info = f"frame_count_config={frame_count_config}"
        loops_info = f"max_b_loops={max_b_loops}"
        if max_b_loops_config:
            loops_info = f"max_b_loops_config={max_b_loops_config}"

        self.logger.info("=" * 80)
        self.logger.info("Running Text2Video Simulation")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, {loops_info}, {frame_info}, "
                        f"strategies={strategies}, portion_stats={portion_stats}, "
                        f"max_sleep_time={max_sleep_time}s")

        script_path = self.workspace_dir / "type1_text2video" / "simulation" / "test_workflow_sim.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--max-b-loops", str(max_b_loops)])
        args.extend(["--frame-count", str(frame_count)])
        args.extend(["--max-sleep-time", str(max_sleep_time)])

        if frame_count_config:
            args.extend(["--frame-count-config", frame_count_config])
        if frame_count_seed is not None:
            args.extend(["--frame-count-seed", str(frame_count_seed)])
        if max_b_loops_config:
            args.extend(["--max-b-loops-config", max_b_loops_config])
        if max_b_loops_seed is not None:
            args.extend(["--max-b-loops-seed", str(max_b_loops_seed)])

        return self._run_script(script_path, args, "Text2Video Simulation")

    def run_text2video_real(self,
                            num_workflows: int = 10,
                            qps: float = 2.0,
                            seed: int = 42,
                            strategies: str = "all",
                            warmup: float = 0.2,
                            duration: int = 120,
                            max_b_loops: int = 3,
                            frame_count: int = 16,
                            frame_count_config: Optional[str] = None,
                            frame_count_seed: Optional[int] = None,
                            max_b_loops_config: Optional[str] = None,
                            max_b_loops_seed: Optional[int] = None,
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
            frame_count: Frame count for video generation (default: 16)
            frame_count_config: Path to JSON config for frame_count distribution (optional)
            frame_count_seed: Random seed for frame_count distribution (optional)
            max_b_loops_config: Path to JSON config for max_b_loops distribution (optional)
            max_b_loops_seed: Random seed for max_b_loops distribution (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        # Build distribution info string
        frame_info = f"frame_count={frame_count}"
        if frame_count_config:
            frame_info = f"frame_count_config={frame_count_config}"
        loops_info = f"max_b_loops={max_b_loops}"
        if max_b_loops_config:
            loops_info = f"max_b_loops_config={max_b_loops_config}"

        self.logger.info("=" * 80)
        self.logger.info("Running Text2Video Real Cluster Mode")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, {loops_info}, {frame_info}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type1_text2video" / "real" / "test_workflow_real.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--max-b-loops", str(max_b_loops)])
        args.extend(["--frame-count", str(frame_count)])

        if frame_count_config:
            args.extend(["--frame-count-config", frame_count_config])
        if frame_count_seed is not None:
            args.extend(["--frame-count-seed", str(frame_count_seed)])
        if max_b_loops_config:
            args.extend(["--max-b-loops-config", max_b_loops_config])
        if max_b_loops_seed is not None:
            args.extend(["--max-b-loops-seed", str(max_b_loops_seed)])

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

    def run_ocr_llm_simulation(self,
                                num_workflows: int = 10,
                                qps: float = 2.0,
                                seed: int = 42,
                                strategies: str = "all",
                                warmup: float = 0.2,
                                duration: int = 120,
                                sleep_time_a_config: Optional[str] = None,
                                sleep_time_b_config: Optional[str] = None,
                                sleep_time_seed: int = 42,
                                portion_stats: float = 1.0) -> Dict:
        """
        Run OCR+LLM workflow in simulation mode.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            sleep_time_a_config: Path to JSON config for A (OCR) sleep time distribution (optional)
            sleep_time_b_config: Path to JSON config for B (LLM) sleep time distribution (optional)
            sleep_time_seed: Random seed for sleep time distribution (default: 42)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running OCR+LLM Simulation")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type4_ocr_llm" / "simulation" / "test_workflow_sim.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)

        if sleep_time_a_config:
            args.extend(["--sleep-time-a-config", sleep_time_a_config])
        if sleep_time_b_config:
            args.extend(["--sleep-time-b-config", sleep_time_b_config])
        if sleep_time_seed is not None:
            args.extend(["--sleep-time-seed", str(sleep_time_seed)])

        return self._run_script(script_path, args, "OCR+LLM Simulation")

    def run_ocr_llm_real(self,
                          num_workflows: int = 10,
                          qps: float = 2.0,
                          seed: int = 42,
                          strategies: str = "all",
                          warmup: float = 0.2,
                          duration: int = 120,
                          image_dir: Optional[str] = None,
                          image_json: Optional[str] = None,
                          ocr_languages: str = "en",
                          ocr_detail_level: str = "standard",
                          max_tokens: int = 512,
                          scheduler_a_url: Optional[str] = None,
                          scheduler_b_url: Optional[str] = None,
                          portion_stats: float = 1.0) -> Dict:
        """
        Run OCR+LLM workflow in real cluster mode.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            image_dir: Directory containing images for OCR (optional)
            image_json: JSON file containing base64-encoded images (optional)
            ocr_languages: OCR languages (default: "en")
            ocr_detail_level: OCR detail level (default: "standard")
            max_tokens: Maximum tokens for LLM generation (default: 512)
            scheduler_a_url: URL for Scheduler A (OCR) (optional)
            scheduler_b_url: URL for Scheduler B (LLM) (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        self.logger.info("=" * 80)
        self.logger.info("Running OCR+LLM Real Cluster Mode")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type4_ocr_llm" / "real" / "test_workflow_real.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)

        if image_dir:
            args.extend(["--image-dir", image_dir])
        if image_json:
            args.extend(["--image-json", image_json])
        args.extend(["--ocr-languages", ocr_languages])
        args.extend(["--ocr-detail-level", ocr_detail_level])
        args.extend(["--max-tokens", str(max_tokens)])
        if scheduler_a_url:
            args.extend(["--scheduler-a-url", scheduler_a_url])
        if scheduler_b_url:
            args.extend(["--scheduler-b-url", scheduler_b_url])

        return self._run_script(script_path, args, "OCR+LLM Real")

    def run_text2image_video_simulation(self,
                                         num_workflows: int = 10,
                                         qps: float = 2.0,
                                         seed: int = 42,
                                         strategies: str = "all",
                                         warmup: float = 0.2,
                                         duration: int = 120,
                                         max_b_loops: int = 3,
                                         frame_count: int = 16,
                                         frame_count_config: Optional[str] = None,
                                         frame_count_seed: Optional[int] = None,
                                         max_b_loops_config: Optional[str] = None,
                                         max_b_loops_seed: Optional[int] = None,
                                         resolution: int = 512,
                                         resolution_config: Optional[str] = None,
                                         resolution_seed: Optional[int] = None,
                                         portion_stats: float = 1.0) -> Dict:
        """
        Run Text2Image+Video workflow in simulation mode.

        This workflow follows the pattern: LLM (A) → FLUX (C) → T2VID (B loops)
        using three separate schedulers.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            max_b_loops: Maximum B task iterations (default: 3)
            frame_count: Frame count for video generation (default: 16)
            frame_count_config: Path to JSON config for frame_count distribution (optional)
            frame_count_seed: Random seed for frame_count distribution (optional)
            max_b_loops_config: Path to JSON config for max_b_loops distribution (optional)
            max_b_loops_seed: Random seed for max_b_loops distribution (optional)
            resolution: Image resolution for FLUX (512 or 1024, default: 512)
            resolution_config: Path to JSON config for resolution distribution (optional)
            resolution_seed: Random seed for resolution distribution (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        # Build distribution info strings
        frame_info = f"frame_count={frame_count}"
        if frame_count_config:
            frame_info = f"frame_count_config={frame_count_config}"
        loops_info = f"max_b_loops={max_b_loops}"
        if max_b_loops_config:
            loops_info = f"max_b_loops_config={max_b_loops_config}"
        res_info = f"resolution={resolution}"
        if resolution_config:
            res_info = f"resolution_config={resolution_config}"

        self.logger.info("=" * 80)
        self.logger.info("Running Text2Image+Video Simulation (A→C→B)")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, {loops_info}, {frame_info}, {res_info}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type3_text2image_video" / "simulation" / "test_workflow_sim.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--max-b-loops", str(max_b_loops)])
        args.extend(["--frame-count", str(frame_count)])
        args.extend(["--resolution", str(resolution)])

        if frame_count_config:
            args.extend(["--frame-count-config", frame_count_config])
        if frame_count_seed is not None:
            args.extend(["--frame-count-seed", str(frame_count_seed)])
        if max_b_loops_config:
            args.extend(["--max-b-loops-config", max_b_loops_config])
        if max_b_loops_seed is not None:
            args.extend(["--max-b-loops-seed", str(max_b_loops_seed)])
        if resolution_config:
            args.extend(["--resolution-config", resolution_config])
        if resolution_seed is not None:
            args.extend(["--resolution-seed", str(resolution_seed)])

        return self._run_script(script_path, args, "Text2Image+Video Simulation")

    def run_text2image_video_real(self,
                                   num_workflows: int = 10,
                                   qps: float = 2.0,
                                   seed: int = 42,
                                   strategies: str = "all",
                                   warmup: float = 0.2,
                                   duration: int = 120,
                                   max_b_loops: int = 3,
                                   frame_count: int = 16,
                                   frame_count_config: Optional[str] = None,
                                   frame_count_seed: Optional[int] = None,
                                   max_b_loops_config: Optional[str] = None,
                                   max_b_loops_seed: Optional[int] = None,
                                   resolution: int = 512,
                                   resolution_config: Optional[str] = None,
                                   resolution_seed: Optional[int] = None,
                                   portion_stats: float = 1.0) -> Dict:
        """
        Run Text2Image+Video workflow in real cluster mode.

        This workflow follows the pattern: LLM (A) → FLUX (C) → T2VID (B loops)
        using three separate schedulers with real model services.

        Args:
            num_workflows: Number of workflows to run (default: 10)
            qps: Queries per second rate (default: 2.0)
            seed: Random seed for reproducibility (default: 42)
            strategies: Comma-separated strategies or 'all' (default: all)
            warmup: Warmup ratio (0.0-1.0) (default: 0.2)
            duration: Maximum experiment duration in seconds (default: 120)
            max_b_loops: Maximum B task iterations (default: 3)
            frame_count: Frame count for video generation (default: 16)
            frame_count_config: Path to JSON config for frame_count distribution (optional)
            frame_count_seed: Random seed for frame_count distribution (optional)
            max_b_loops_config: Path to JSON config for max_b_loops distribution (optional)
            max_b_loops_seed: Random seed for max_b_loops distribution (optional)
            resolution: Image resolution for FLUX (512 or 1024, default: 512)
            resolution_config: Path to JSON config for resolution distribution (optional)
            resolution_seed: Random seed for resolution distribution (optional)
            portion_stats: Portion of non-warmup workflows for statistics (default: 1.0)

        Returns:
            Dict with experiment results and metrics path
        """
        # Build distribution info strings
        frame_info = f"frame_count={frame_count}"
        if frame_count_config:
            frame_info = f"frame_count_config={frame_count_config}"
        loops_info = f"max_b_loops={max_b_loops}"
        if max_b_loops_config:
            loops_info = f"max_b_loops_config={max_b_loops_config}"
        res_info = f"resolution={resolution}"
        if resolution_config:
            res_info = f"resolution_config={resolution_config}"

        self.logger.info("=" * 80)
        self.logger.info("Running Text2Image+Video Real Cluster Mode (A→C→B)")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: num_workflows={num_workflows}, qps={qps}, "
                        f"duration={duration}s, {loops_info}, {frame_info}, {res_info}, "
                        f"strategies={strategies}, portion_stats={portion_stats}")

        script_path = self.workspace_dir / "type3_text2image_video" / "real" / "test_workflow_real.py"
        args = self._build_common_args(num_workflows, qps, seed, strategies, warmup, duration, portion_stats)
        args.extend(["--max-b-loops", str(max_b_loops)])
        args.extend(["--frame-count", str(frame_count)])
        args.extend(["--resolution", str(resolution)])

        if frame_count_config:
            args.extend(["--frame-count-config", frame_count_config])
        if frame_count_seed is not None:
            args.extend(["--frame-count-seed", str(frame_count_seed)])
        if max_b_loops_config:
            args.extend(["--max-b-loops-config", max_b_loops_config])
        if max_b_loops_seed is not None:
            args.extend(["--max-b-loops-seed", str(max_b_loops_seed)])
        if resolution_config:
            args.extend(["--resolution-config", resolution_config])
        if resolution_seed is not None:
            args.extend(["--resolution-seed", str(resolution_seed)])

        return self._run_script(script_path, args, "Text2Image+Video Real")


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
