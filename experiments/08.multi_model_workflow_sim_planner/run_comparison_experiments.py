#!/usr/bin/env python3
"""
Run Comparison Experiments: Automated Batch Execution

This script automatically runs all 6 experiment configurations:
- 3 strategies (min_time, probabilistic, round_robin)
- 2 modes (with/without migration)

For each configuration, it:
1. Stops all services
2. Archives previous experiment's logs and intermediate results
3. Cleans up temporary files (service logs, PID files)
4. Starts all services
5. Runs the experiment (using new /model/restart API for migrations)
6. Collects results

Flexible Instance Configuration:
- Configurable number of test instances via --total-instances (default: 16)
- Configurable port range via --instance-start-port (default: 8210)
- Instances will use ports [start_port, start_port + total_instances - 1]
- Port range validation ensures no overflow beyond 65535
- Phase configurations are automatically calculated to maintain original ratios:
  * Phase 1: A=25%, B=75% (fanout=3)
  * Phase 2: A=12.5%, B=87.5% (fanout=8)
  * Phase 3: A=50%, B=50% (fanout=1)

Migration Flow (using instance /model/restart API):
- Experiments with migration enabled use the new restart API
- Migration runs in background, doesn't block task submission
- Instance handles drain, task completion, stop, deregister, start, register internally
- Background monitor polls status at 500ms intervals

Log Management:
- Experiment logs are archived to archived_logs/<experiment_name>_<timestamp>/
- Service logs (scheduler, instance, predictor) are cleaned between runs
- Intermediate results and key logs are preserved for each sub-experiment

Usage Examples:
  # Default: 16 instances on ports 8210-8225
  # Phase configs: P1(A=4,B=12), P2(A=2,B=14), P3(A=8,B=8)
  python3 run_comparison_experiments.py

  # Small scale: 8 instances on ports 9000-9007
  # Phase configs: P1(A=2,B=6), P2(A=1,B=7), P3(A=4,B=4)
  python3 run_comparison_experiments.py --total-instances 8 --instance-start-port 9000

  # Large scale: 32 instances with higher QPS
  # Phase configs: P1(A=8,B=24), P2(A=4,B=28), P3(A=16,B=16)
  python3 run_comparison_experiments.py --total-instances 32 --qps 5.0

  # Quick test: 4 instances with fewer workflows
  # Phase configs: P1(A=1,B=3), P2(A=1,B=3), P3(A=2,B=2)
  python3 run_comparison_experiments.py --total-instances 4 --num-workflows-per-phase 20

Author: Claude Code
Date: 2025-11-03
"""

import argparse
import json
import os
import shutil
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

def calculate_phase_configs(total_instances: int) -> List[tuple]:
    """
    Calculate phase configurations based on total number of instances.

    Maintains the original ratios from the 16-instance baseline:
    - Phase 1: A=25%, B=75% (fanout=3)
    - Phase 2: A=12.5%, B=87.5% (fanout=8)
    - Phase 3: A=50%, B=50% (fanout=1)

    Args:
        total_instances: Total number of instances available

    Returns:
        List of tuples: [(phase_id, fanout, scheduler_a_instances, scheduler_b_instances), ...]
    """
    # Original ratios based on 16 instances
    phase_ratios = [
        (1, 3, 0.25, 0.75),    # Phase 1: A=4, B=12 (fanout=3)
        (2, 8, 0.125, 0.875),  # Phase 2: A=2, B=14 (fanout=8)
        (3, 1, 0.5, 0.5),      # Phase 3: A=8, B=8 (fanout=1)
    ]

    phase_configs = []
    for phase_id, fanout, ratio_a, ratio_b in phase_ratios:
        # Calculate instances for each scheduler
        instances_a = round(total_instances * ratio_a)
        instances_b = round(total_instances * ratio_b)

        # Ensure at least 1 instance per scheduler and total equals total_instances
        instances_a = max(1, instances_a)
        instances_b = max(1, instances_b)

        # Adjust if rounding caused mismatch
        current_total = instances_a + instances_b
        if current_total != total_instances:
            # Adjust the larger scheduler to match total
            if instances_b > instances_a:
                instances_b = total_instances - instances_a
            else:
                instances_a = total_instances - instances_b

        phase_configs.append((phase_id, fanout, instances_a, instances_b))

    return phase_configs


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    strategy: str
    enable_migration: bool
    num_workflows_per_phase: int
    qps: float
    total_instances: int
    instance_start_port: int
    phase_configs: List[tuple] = field(default_factory=list)

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

        # Add phase configurations as comma-separated string
        # Format: "phase_id:fanout:a_instances:b_instances,..."
        if self.phase_configs:
            phase_str = ",".join(
                f"{pid}:{fan}:{a_inst}:{b_inst}"
                for pid, fan, a_inst, b_inst in self.phase_configs
            )
            args.extend(["--phase-configs", phase_str])

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

    def clean_temporary_files(self, preserve_experiment_name: Optional[str] = None):
        """
        Clean up temporary log files and PIDs, optionally preserving experiment logs.

        Args:
            preserve_experiment_name: If provided, archive logs for this experiment before cleanup
        """
        logger.info("Cleaning temporary files...")

        # If we need to preserve experiment logs, archive them first
        if preserve_experiment_name and self.logs_dir.exists():
            self._archive_experiment_logs(preserve_experiment_name)

        # Remove service log files (scheduler, instance, predictor logs)
        if self.logs_dir.exists():
            service_log_patterns = [
                "scheduler-*.log",
                "instance-*.log",
                "predictor.log",
                "experiment_*.log"  # Also remove experiment logs after archiving
            ]

            for pattern in service_log_patterns:
                for log_file in self.logs_dir.glob(pattern):
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

    def _archive_experiment_logs(self, experiment_name: str):
        """
        Archive logs and intermediate results for a completed experiment.

        Args:
            experiment_name: Name of the experiment (e.g., "min_time_migration")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = self.experiment_dir / "archived_logs" / f"{experiment_name}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Archiving experiment logs to {archive_dir.relative_to(self.experiment_dir)}...")

        # Archive experiment script logs
        experiment_logs = list(self.logs_dir.glob("experiment_*.log"))
        for log_file in experiment_logs:
            try:
                dest = archive_dir / log_file.name
                shutil.copy2(log_file, dest)
                logger.debug(f"  Archived {log_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to archive {log_file.name}: {e}")

        # Archive key service logs (scheduler logs contain important scheduling decisions)
        important_service_logs = [
            "scheduler-a.log",
            "scheduler-b.log",
            "predictor.log"
        ]
        for log_name in important_service_logs:
            log_file = self.logs_dir / log_name
            if log_file.exists():
                try:
                    dest = archive_dir / log_name
                    shutil.copy2(log_file, dest)
                    logger.debug(f"  Archived {log_name}")
                except Exception as e:
                    logger.warning(f"  Failed to archive {log_name}: {e}")

        # Create a metadata file
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "archived_at": datetime.now().isoformat(),
        }
        metadata_file = archive_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"  Failed to write metadata: {e}")

        logger.info(f"  ✓ Archived {len(experiment_logs)} experiment logs")

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
        total_instances: int = 16,
        instance_start_port: int = 8210,
        output_dir: Optional[Path] = None,
        experiment_timeout: int = 900,
    ):
        self.experiment_dir = experiment_dir
        self.num_workflows_per_phase = num_workflows_per_phase
        self.qps = qps
        self.total_instances = total_instances
        self.instance_start_port = instance_start_port
        self.output_dir = output_dir or (experiment_dir / "results")
        self.experiment_timeout = experiment_timeout

        self.service_manager = ServiceManager(experiment_dir)
        self.results: List[ExperimentResult] = []

    def generate_configurations(self) -> List[ExperimentConfig]:
        """Generate all 6 experiment configurations with dynamic phase configs."""
        strategies = ["min_time", "probabilistic", "round_robin"]
        migration_modes = [True, False]

        # Calculate phase configurations based on total instances
        phase_configs = calculate_phase_configs(self.total_instances)

        logger.info(f"Calculated phase configurations for {self.total_instances} instances:")
        for phase_id, fanout, a_inst, b_inst in phase_configs:
            logger.info(f"  Phase {phase_id}: fanout={fanout}, A={a_inst}, B={b_inst} (total={a_inst+b_inst})")

        configs = []
        for strategy in strategies:
            for enable_migration in migration_modes:
                config = ExperimentConfig(
                    strategy=strategy,
                    enable_migration=enable_migration,
                    num_workflows_per_phase=self.num_workflows_per_phase,
                    qps=self.qps,
                    total_instances=self.total_instances,
                    instance_start_port=self.instance_start_port,
                    phase_configs=phase_configs,
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
                    timeout=self.experiment_timeout
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
            result.error_message = f"Timeout (>{self.experiment_timeout}s)"
            logger.error(f"✗ Experiment timed out (>{self.experiment_timeout}s)")
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
        logger.info(f"Total instances: {self.total_instances}")
        logger.info(f"Instance port range: {self.instance_start_port}-{self.instance_start_port + self.total_instances - 1}")
        logger.info(f"Output directory: {self.output_dir}\n")

        previous_experiment_name = None

        for idx, config in enumerate(configs, 1):
            logger.info(f"\n[{idx}/{len(configs)}] Preparing: {config.get_name()}")

            # Stop services
            if not self.service_manager.stop_all_services():
                logger.error("Failed to stop services, continuing anyway...")

            # Clean up and archive previous experiment's logs
            self.service_manager.clean_temporary_files(preserve_experiment_name=previous_experiment_name)

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
                previous_experiment_name = config.get_name()  # Track even if failed
                continue

            # Run experiment
            result = self.run_single_experiment(config)
            self.results.append(result)

            # Track this experiment for next iteration's archive
            previous_experiment_name = config.get_name()

            # Brief pause between experiments
            time.sleep(2)

        # Final cleanup - archive the last experiment's logs
        logger.info("\nFinal cleanup...")
        self.service_manager.stop_all_services()
        if previous_experiment_name:
            self.service_manager.clean_temporary_files(preserve_experiment_name=previous_experiment_name)

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
                "total_instances": self.total_instances,
                "instance_start_port": self.instance_start_port,
                "instance_port_range": f"{self.instance_start_port}-{self.instance_start_port + self.total_instances - 1}",
                "experiment_timeout": self.experiment_timeout,
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
        description="Run all comparison experiments (6 configurations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (16 instances on ports 8210-8225)
  python3 run_comparison_experiments.py

  # Run with 8 instances on ports 9000-9007
  python3 run_comparison_experiments.py --total-instances 8 --instance-start-port 9000

  # Run with custom workflow count and QPS
  python3 run_comparison_experiments.py --num-workflows-per-phase 200 --qps 5.0

  # Run with all custom settings
  python3 run_comparison_experiments.py \\
    --total-instances 32 \\
    --instance-start-port 8300 \\
    --num-workflows-per-phase 150 \\
    --qps 3.0
        """
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
        "--total-instances",
        type=int,
        default=16,
        help="Total number of model instances to use (default: 16). "
             "Phase configurations will be automatically calculated to maintain the original ratios: "
             "P1(A=25%%, B=75%%), P2(A=12.5%%, B=87.5%%), P3(A=50%%, B=50%%)"
    )
    parser.add_argument(
        "--instance-start-port",
        type=int,
        default=8210,
        help="Starting port for instance range (default: 8210). Instances will use ports [start, start+total-1]"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./results)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds for each experiment (default: 900)"
    )

    args = parser.parse_args()

    # Validate instance configuration
    if args.total_instances < 1:
        parser.error("--total-instances must be at least 1")
    if args.instance_start_port < 1024 or args.instance_start_port > 65535:
        parser.error("--instance-start-port must be in range [1024, 65535]")

    end_port = args.instance_start_port + args.total_instances - 1
    if end_port > 65535:
        parser.error(f"Port range [{args.instance_start_port}, {end_port}] exceeds maximum port 65535. "
                    f"Reduce --total-instances or lower --instance-start-port")

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
        total_instances=args.total_instances,
        instance_start_port=args.instance_start_port,
        output_dir=output_dir,
        experiment_timeout=args.timeout,
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
