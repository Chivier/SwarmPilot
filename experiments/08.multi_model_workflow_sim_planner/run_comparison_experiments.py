#!/usr/bin/env python3
"""
Run Comparison Experiments: Automated Batch Execution

This script automatically runs all 6 experiment configurations:
- 3 strategies (min_time, probabilistic, round_robin)
- 2 modes (with/without migration)

For each configuration, it:
1. Stops all services
2. Cleans up temporary files
3. Starts all services
4. Runs the experiment
5. Collects results

Author: Claude Code
Date: 2025-11-03
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    strategy: str
    enable_migration: bool
    num_workflows_per_phase: int
    qps: float
    total_instances: int
    instance_start_port: int

    def get_name(self) -> str:
        """Get a descriptive name for this configuration."""
        mode = "migration" if self.enable_migration else "static"
        return f"{self.strategy}_{mode}"

    def get_command_args(self) -> List[str]:
        """Get command line arguments for test_dynamic_migration.py."""
        args = [
            "uv", "run", "python3", "test_dynamic_migration.py",
            "--strategy", self.strategy,
            "--num-workflows-per-phase", str(self.num_workflows_per_phase),
            "--qps", str(self.qps),
            "--total-instances", str(self.total_instances),
            "--instance-start-port", str(self.instance_start_port),
        ]

        if self.enable_migration:
            args.append("--enable-migration")
        else:
            args.append("--disable-migration")

        return args


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    config: ExperimentConfig
    success: bool
    start_time: datetime
    end_time: Optional[datetime] = None
    result_file: Optional[str] = None
    error_message: Optional[str] = None

    def get_duration_seconds(self) -> Optional[float]:
        """Get experiment duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# ============================================================================
# Service Management
# ============================================================================

class ServiceManager:
    """Manages starting and stopping services."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.start_script = experiment_dir / "start_all_services.sh"
        self.stop_script = experiment_dir / "stop_all_services.sh"
        self.logs_dir = experiment_dir / "logs"

    def stop_all_services(self) -> bool:
        """Stop all running services."""
        logger.info("Stopping all services...")
        try:
            result = subprocess.run(
                ["bash", str(self.stop_script)],
                cwd=self.experiment_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info("✓ All services stopped")
                return True
            else:
                logger.warning(f"Stop script returned code {result.returncode}")
                logger.warning(f"stderr: {result.stderr}")
                return True  # Continue anyway
        except subprocess.TimeoutExpired:
            logger.error("Timeout stopping services")
            return False
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
            return False

    def clean_temporary_files(self):
        """Clean up temporary log files and PIDs."""
        logger.info("Cleaning temporary files...")

        # Remove log files
        if self.logs_dir.exists():
            for log_file in self.logs_dir.glob("*.log"):
                try:
                    log_file.unlink()
                    logger.debug(f"Removed {log_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {log_file.name}: {e}")

            # Remove PID files
            for pid_file in self.logs_dir.glob("*.pid"):
                try:
                    pid_file.unlink()
                    logger.debug(f"Removed {pid_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove {pid_file.name}: {e}")

        logger.info("✓ Cleanup complete")

    def start_all_services(self, timeout: int = 120) -> bool:
        """Start all services and wait for health checks."""
        logger.info("Starting all services...")

        try:
            result = subprocess.run(
                ["bash", str(self.start_script)],
                cwd=self.experiment_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                logger.info("✓ All services started and healthy")
                return True
            else:
                logger.error(f"Start script failed with code {result.returncode}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout starting services (>{timeout}s)")
            return False
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return False

    def wait_for_port_release(self, seconds: int = 2):
        """Wait for ports to be released after stopping services."""
        logger.info(f"Waiting {seconds}s for ports to be released...")
        time.sleep(seconds)


# ============================================================================
# Experiment Runner
# ============================================================================

class ComparisonExperimentRunner:
    """Runs all comparison experiments."""

    def __init__(
        self,
        experiment_dir: Path,
        num_workflows_per_phase: int,
        qps: float,
        output_dir: Optional[Path] = None,
    ):
        self.experiment_dir = experiment_dir
        self.num_workflows_per_phase = num_workflows_per_phase
        self.qps = qps
        self.output_dir = output_dir or (experiment_dir / "results")

        self.service_manager = ServiceManager(experiment_dir)
        self.results: List[ExperimentResult] = []

    def generate_configurations(self) -> List[ExperimentConfig]:
        """Generate all 6 experiment configurations."""
        strategies = ["min_time", "probabilistic", "round_robin"]
        migration_modes = [True, False]

        configs = []
        for strategy in strategies:
            for enable_migration in migration_modes:
                config = ExperimentConfig(
                    strategy=strategy,
                    enable_migration=enable_migration,
                    num_workflows_per_phase=self.num_workflows_per_phase,
                    qps=self.qps,
                    total_instances=16,
                    instance_start_port=8210,
                )
                configs.append(config)

        return configs

    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment configuration."""
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {config.get_name()}")
        logger.info(f"{'='*70}\n")

        result = ExperimentResult(
            config=config,
            success=False,
            start_time=datetime.now(),
        )

        try:
            # Run experiment
            cmd = config.get_command_args()
            logger.info(f"Command: {' '.join(cmd)}")

            log_file = self.experiment_dir / "logs" / f"experiment_{config.get_name()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    cwd=self.experiment_dir,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=900  # 15 minutes max
                )

            result.end_time = datetime.now()

            if process.returncode == 0:
                logger.info(f"✓ Experiment completed successfully")
                result.success = True

                # Find result file
                result_files = list(self.output_dir.glob(f"exp08_{config.get_name().split('_')[1]}_{config.strategy}_*.json"))
                if result_files:
                    # Get the most recent file
                    result.result_file = str(sorted(result_files, key=lambda x: x.stat().st_mtime)[-1])
                    logger.info(f"  Result file: {Path(result.result_file).name}")
            else:
                logger.error(f"✗ Experiment failed with code {process.returncode}")
                result.error_message = f"Exit code: {process.returncode}"

            logger.info(f"  Log file: {log_file.name}")
            logger.info(f"  Duration: {result.get_duration_seconds():.1f}s")

        except subprocess.TimeoutExpired:
            result.end_time = datetime.now()
            result.error_message = "Timeout (>15 minutes)"
            logger.error(f"✗ Experiment timed out")
        except Exception as e:
            result.end_time = datetime.now()
            result.error_message = str(e)
            logger.error(f"✗ Experiment error: {e}")

        return result

    def run_all_experiments(self):
        """Run all experiment configurations."""
        configs = self.generate_configurations()

        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Comparison Experiment Suite")
        logger.info(f"{'='*70}\n")
        logger.info(f"Total configurations: {len(configs)}")
        logger.info(f"Workflows per phase: {self.num_workflows_per_phase}")
        logger.info(f"QPS: {self.qps}")
        logger.info(f"Output directory: {self.output_dir}\n")

        for idx, config in enumerate(configs, 1):
            logger.info(f"\n[{idx}/{len(configs)}] Preparing: {config.get_name()}")

            # Stop services
            if not self.service_manager.stop_all_services():
                logger.error("Failed to stop services, continuing anyway...")

            # Clean up
            self.service_manager.clean_temporary_files()

            # Wait for ports to be released
            self.service_manager.wait_for_port_release()

            # Start services
            if not self.service_manager.start_all_services():
                logger.error(f"Failed to start services for {config.get_name()}, skipping...")
                result = ExperimentResult(
                    config=config,
                    success=False,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message="Failed to start services"
                )
                self.results.append(result)
                continue

            # Run experiment
            result = self.run_single_experiment(config)
            self.results.append(result)

            # Brief pause between experiments
            time.sleep(2)

        # Final cleanup
        logger.info("\nFinal cleanup...")
        self.service_manager.stop_all_services()

        # Print summary
        self.print_summary()

        # Save summary
        self.save_summary()

    def print_summary(self):
        """Print summary of all experiments."""
        logger.info(f"\n{'='*70}")
        logger.info("Experiment Suite Summary")
        logger.info(f"{'='*70}\n")

        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful

        logger.info(f"Total experiments: {len(self.results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}\n")

        for result in self.results:
            status = "✓" if result.success else "✗"
            duration = f"{result.get_duration_seconds():.1f}s" if result.get_duration_seconds() else "N/A"
            logger.info(f"{status} {result.config.get_name():30s} - {duration:>8s}")
            if result.error_message:
                logger.info(f"    Error: {result.error_message}")

    def save_summary(self):
        """Save experiment summary to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"comparison_run_{timestamp}.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "num_workflows_per_phase": self.num_workflows_per_phase,
                "qps": self.qps,
            },
            "total_experiments": len(self.results),
            "successful": sum(1 for r in self.results if r.success),
            "failed": sum(1 for r in self.results if not r.success),
            "experiments": []
        }

        for result in self.results:
            exp_data = {
                "name": result.config.get_name(),
                "strategy": result.config.strategy,
                "migration_enabled": result.config.enable_migration,
                "success": result.success,
                "duration_seconds": result.get_duration_seconds(),
                "result_file": result.result_file,
                "error_message": result.error_message,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
            }
            summary["experiments"].append(exp_data)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n✓ Summary saved to: {summary_file}")
        return summary_file


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all comparison experiments (6 configurations)"
    )
    parser.add_argument(
        "--num-workflows-per-phase",
        type=int,
        default=100,
        help="Number of workflows per phase (default: 100)"
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=2.0,
        help="QPS for task submission (default: 2.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./results)"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Get experiment directory
    experiment_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Create and run
    runner = ComparisonExperimentRunner(
        experiment_dir=experiment_dir,
        num_workflows_per_phase=args.num_workflows_per_phase,
        qps=args.qps,
        output_dir=output_dir,
    )

    try:
        runner.run_all_experiments()
        logger.info("\n✅ All experiments completed!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user")
        runner.service_manager.stop_all_services()
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
