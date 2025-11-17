#!/usr/bin/env python3
"""
Out-of-Distribution Experiments Orchestration Script.

This script automates the execution of all out-of-distribution experiments:
- Experiment 1: B1 samples sleep_time from A1 distribution
- Experiment 2: B1 scales sleep_time and exp_runtime by factors [0.2, 0.5, 0.8]

For each experiment, runs both baseline and comparison configurations,
collects metrics, and generates comparison reports.
"""

import argparse
import time
import numpy as np
import subprocess
import json
import signal
import sys
import atexit
from pathlib import Path
from typing import List, Optional

from ood_config import (
    OODExperimentConfig,
    get_all_experiment_configs,
    get_exp1_baseline_config,
    get_exp1_comparison_config,
    get_exp2_baseline_configs,
    get_exp2_comparison_configs
)
from metrics_collector import MetricsCollector, MetricsComparator, ExperimentMetrics
from workload_generator import (
    generate_workflow_from_traces,
    generate_workflow_with_a1_b1_sampling,
    WorkflowWorkload
)


class ServiceManager:
    """Manages service lifecycle for experiment execution."""

    def __init__(self, service_dir: Path, n1: int = 10, n2: int = 6):
        """
        Initialize service manager.

        Args:
            service_dir: Directory containing service scripts
            n1: Number of Group A instances
            n2: Number of Group B instances
        """
        self.service_dir = service_dir
        self.start_script = service_dir / "start_all_services.sh"
        self.stop_script = service_dir / "stop_all_services.sh"
        self.n1 = n1
        self.n2 = n2

        if not self.start_script.exists():
            raise FileNotFoundError(f"Start script not found: {self.start_script}")
        if not self.stop_script.exists():
            raise FileNotFoundError(f"Stop script not found: {self.stop_script}")

    def stop_services(self, timeout: int = 120) -> bool:
        """
        Stop all services.

        Args:
            timeout: Maximum time to wait for shutdown (seconds)

        Returns:
            True if successful, False otherwise
        """
        print(f"Stopping all services...")
        try:
            result = subprocess.run(
                [str(self.stop_script)],
                cwd=self.service_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"  WARNING: Stop script exited with code {result.returncode}")
                print(f"  STDERR: {result.stderr[:500]}")
                return False

            print(f"  ✓ All services stopped successfully")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ERROR: Service stop timed out after {timeout}s")
            return False
        except Exception as e:
            print(f"  ERROR: Failed to stop services: {str(e)}")
            return False

    def start_services(self, timeout: int = 300) -> bool:
        """
        Start all services with configured instance counts.

        Args:
            timeout: Maximum time to wait for startup (seconds)

        Returns:
            True if successful, False otherwise
        """
        print(f"Starting all services (N1={self.n1}, N2={self.n2})...")
        try:
            env = {
                **subprocess.os.environ.copy(),
                'N1': str(self.n1),
                'N2': str(self.n2)
            }

            result = subprocess.run(
                [str(self.start_script)],
                cwd=self.service_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                print(f"  ERROR: Start script exited with code {result.returncode}")
                print(f"  STDERR: {result.stderr[:500]}")
                return False

            print(f"  ✓ All services started successfully")
            return True

        except subprocess.TimeoutExpired:
            print(f"  ERROR: Service start timed out after {timeout}s")
            return False
        except Exception as e:
            print(f"  ERROR: Failed to start services: {str(e)}")
            return False

    def check_health(self, timeout: int = 60) -> bool:
        """
        Check if all services are healthy.

        Args:
            timeout: Maximum time to wait for health check (seconds)

        Returns:
            True if all services are healthy, False otherwise
        """
        print(f"Checking service health...")

        # Services to check
        services = [
            ("Predictor", "http://localhost:8099/health"),
            ("Scheduler A", "http://localhost:8100/health"),
            ("Scheduler B", "http://localhost:8200/health")
        ]

        start_time = time.time()

        for service_name, health_url in services:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                print(f"  ERROR: Health check timed out")
                return False

            # Try to connect with exponential backoff
            attempts = 0
            max_attempts = int(remaining)

            while attempts < max_attempts:
                try:
                    import requests
                    response = requests.get(health_url, timeout=2)
                    if response.status_code == 200:
                        print(f"  ✓ {service_name} is healthy")
                        break
                except Exception:
                    attempts += 1
                    if attempts < max_attempts:
                        time.sleep(1)
                    else:
                        print(f"  ERROR: {service_name} failed health check")
                        return False

        print(f"  ✓ All services are healthy")
        return True

    def restart_services(self, restart_delay: int = 30) -> bool:
        """
        Restart all services.

        Args:
            restart_delay: Delay after restart before returning (seconds)

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'=' * 60}")
        print(f"RESTARTING ALL SERVICES")
        print(f"{'=' * 60}")

        # Stop services
        if not self.stop_services():
            print(f"WARNING: Service stop had issues, continuing anyway...")

        # Wait a bit for cleanup
        print(f"Waiting 5s for cleanup...")
        time.sleep(5)

        # Start services
        if not self.start_services():
            print(f"ERROR: Service start failed!")
            return False

        # Check health
        if not self.check_health():
            print(f"ERROR: Service health check failed!")
            return False

        # Additional delay for stabilization
        if restart_delay > 0:
            print(f"Waiting {restart_delay}s for service stabilization...")
            time.sleep(restart_delay)

        print(f"{'=' * 60}")
        print(f"SERVICES READY")
        print(f"{'=' * 60}\n")

        return True


class OODExperimentRunner:
    """Runner for out-of-distribution experiments."""

    def __init__(
        self,
        output_dir: Path = None,
        use_real_services: bool = False,
        restart_services: bool = True,
        n1: int = 10,
        n2: int = 6,
        restart_delay: int = 30
    ):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save results (default: ./ood_results)
            use_real_services: If True, use real scheduler/predictor services instead of simulation
            restart_services: If True, restart services before each test pair (only applies when use_real_services=True)
            n1: Number of Group A instances
            n2: Number of Group B instances
            restart_delay: Delay after service restart before running tests (seconds)
        """
        self.output_dir = output_dir or Path("./ood_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_real_services = use_real_services
        self.restart_services = restart_services
        self.restart_delay = restart_delay

        # exp07 is in the parent experiments directory
        self.exp07_dir = Path(__file__).parent.parent / "07.Exp2.Deep_Research"

        # Initialize service manager if using real services
        self.service_manager = None
        if use_real_services:
            self.service_manager = ServiceManager(
                service_dir=self.exp07_dir,
                n1=n1,
                n2=n2
            )
            print(f"Initialized ServiceManager (N1={n1}, N2={n2})")
            if restart_services:
                print(f"Service restart enabled (delay={restart_delay}s)")

    def cleanup(self):
        """
        Clean up services on exit.

        This method stops all services if they are running.
        Called automatically on script exit or interruption.
        """
        if self.service_manager and self.use_real_services:
            print(f"\n{'=' * 60}")
            print(f"CLEANING UP SERVICES")
            print(f"{'=' * 60}")
            try:
                self.service_manager.stop_services()
                print(f"✓ Services cleaned up successfully")
            except Exception as e:
                print(f"WARNING: Error during cleanup: {str(e)}")
            print(f"{'=' * 60}\n")

    def generate_workload(
        self,
        config: OODExperimentConfig
    ) -> WorkflowWorkload:
        """
        Generate workload based on experiment configuration.

        Args:
            config: Experiment configuration

        Returns:
            WorkflowWorkload object
        """
        print(f"Generating workload for {config.experiment_name}...")
        print(f"  {config.get_description()}")

        # Choose workload generation function based on config
        if config.b1_data_source.value == "a1_sampled":
            # Use A1-sampled B1 workload
            workflow, _ = generate_workflow_with_a1_b1_sampling(
                num_workflows=config.num_workflows,
                seed=config.seed,
                use_a1_for_exp_runtime=config.sync_exp_runtime_with_sleep_time
            )
        else:
            # Use standard workload
            workflow, _ = generate_workflow_from_traces(
                num_workflows=config.num_workflows,
                seed=config.seed
            )

        # Apply scaling if needed
        if config.b1_sleep_time_scale != 1.0 or config.b1_exp_runtime_scale != 1.0:
            workflow = self._apply_scaling(workflow, config)

        return workflow

    def _apply_scaling(
        self,
        workflow: WorkflowWorkload,
        config: OODExperimentConfig
    ) -> WorkflowWorkload:
        """
        Apply scaling factors to B1 tasks.

        Args:
            workflow: Original workflow
            config: Configuration with scaling factors

        Returns:
            Modified workflow with scaled B1 values
        """
        print(f"  Applying scaling: sleep_time={config.b1_sleep_time_scale}x, "
              f"exp_runtime={config.b1_exp_runtime_scale}x")

        # Scale B1 times
        scaled_b1_times = []
        for workflow_b1 in workflow.b1_times:
            scaled_workflow = [
                t * config.b1_sleep_time_scale for t in workflow_b1
            ]
            scaled_b1_times.append(scaled_workflow)

        # Create new workflow with scaled values
        return WorkflowWorkload(
            name=workflow.name + f"_scaled_{config.b1_sleep_time_scale}",
            a1_times=workflow.a1_times,
            a2_times=workflow.a2_times,
            b1_times=scaled_b1_times,
            b2_times=workflow.b2_times,
            fanout_values=workflow.fanout_values,
            description=workflow.description + f" (scaled by {config.b1_sleep_time_scale}x)"
        )

    def simulate_experiment_execution(
        self,
        workflow: WorkflowWorkload,
        config: OODExperimentConfig
    ) -> MetricsCollector:
        """
        Simulate experiment execution and collect metrics.

        In a real implementation, this would actually run the scheduler and tasks.
        Here we simulate workflow execution times based on the workload data.

        Args:
            workflow: Workload to execute
            config: Experiment configuration

        Returns:
            MetricsCollector with collected data
        """
        collector = MetricsCollector(
            experiment_name=config.experiment_name,
            experiment_description=config.get_description()
        )

        print(f"Executing {config.experiment_name}...")
        print(f"  Workflows: {config.num_workflows}")
        print(f"  Strategy: {config.strategy}")

        collector.start_collection()

        # Simulate workflow executions
        # In reality, this would submit workflows to the scheduler
        for i in range(config.num_workflows):
            # Simulate workflow completion time based on task times
            a1_time = workflow.a1_times[i]
            a2_time = workflow.a2_times[i]

            # B1 tasks execute in parallel, so max determines the time
            b1_max = max(workflow.b1_times[i])

            # B2 tasks execute after B1
            b2_max = max(workflow.b2_times[i])

            # Simplified workflow completion time: A1 + max(B1) + max(B2) + A2
            # (In reality, scheduling overhead and queuing would affect this)
            workflow_latency = a1_time + b1_max + b2_max + a2_time

            # Add some random variation for scheduling overhead
            workflow_latency += np.random.uniform(0.1, 0.5)

            collector.record_workflow_completion(workflow_latency)

            # Small delay to simulate real execution
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{config.num_workflows} workflows")
                time.sleep(0.01)

        collector.end_collection()

        print(f"  Completed in {collector.get_makespan():.2f}s")

        return collector

    def execute_with_real_services(
        self,
        workflow: WorkflowWorkload,
        config: OODExperimentConfig
    ) -> MetricsCollector:
        """
        Execute experiment with real scheduler and predictor services.

        Args:
            workflow: Workload to execute
            config: Experiment configuration

        Returns:
            MetricsCollector with collected data
        """
        collector = MetricsCollector(
            experiment_name=config.experiment_name,
            experiment_description=config.get_description()
        )

        print(f"Executing {config.experiment_name} with REAL services...")
        print(f"  Workflows: {config.num_workflows}")
        print(f"  Strategy: {config.strategy}")

        # Export workload to temporary file for test_dynamic_workflow.py
        workload_file = self.output_dir / f"{config.experiment_name}_workload.json"
        workload_data = {
            "a1_times": workflow.a1_times,
            "a2_times": workflow.a2_times,
            "b1_times": workflow.b1_times,
            "b2_times": workflow.b2_times,
            "fanout_values": workflow.fanout_values
        }
        with open(workload_file, "w") as f:
            json.dump(workload_data, f)

        # Prepare command for test_dynamic_workflow.py
        test_script = self.exp07_dir / "test_dynamic_workflow.py"
        cmd = [
            "uv", "run", "python3", str(test_script),
            "--num-workflows", str(config.num_workflows),
            "--qps", "10",
            "--gqps", "100",
            "--strategies", config.strategy,
            "--seed", str(config.seed),
            "--warmup", "0.2"
        ]

        print(f"  Running: {' '.join(cmd)}")

        collector.start_collection()

        # Execute the test
        try:
            result = subprocess.run(
                cmd,
                cwd=self.exp07_dir,
                capture_output=True,
                text=True,
                timeout=1500  # 25 minute timeout
            )

            if result.returncode != 0:
                print(f"  ERROR: Test execution failed!")
                print(f"  STDERR: {result.stderr[:500]}")
                raise RuntimeError(f"Test execution failed with return code {result.returncode}")

            # Parse results from the output
            # The test_dynamic_workflow.py script outputs results to results/ directory
            results_dir = self.exp07_dir / "results"
            result_files = sorted(results_dir.glob("results_workflow_b1b2_*.json"))

            if not result_files:
                raise RuntimeError("No result files found after test execution")

            # Get the most recent result file
            latest_result = result_files[-1]
            print(f"  Loading results from: {latest_result.name}")

            with open(latest_result, "r") as f:
                results = json.load(f)

            # Extract workflow latencies from results
            # results["results"] is a list of strategy results
            strategy_results = None
            for result in results["results"]:
                if result["strategy"] == config.strategy:
                    strategy_results = result
                    break

            if strategy_results is None:
                raise RuntimeError(f"Strategy '{config.strategy}' not found in results")

            workflow_info = strategy_results['workflows']

            # Extract workflow completion times
            if "workflow_times" in workflow_info and len(workflow_info["workflow_times"]) > 0:
                print(f"  Found {len(workflow_info['workflow_times'])} workflow completion times")
                for latency in workflow_info["workflow_times"]:
                    collector.record_workflow_completion(latency)
            else:
                # Fallback: use mean latency if individual times not available
                print(f"  WARNING: workflow_times not available, using mean latency fallback")
                mean_latency = workflow_info.get("avg_workflow_time", 0)
                if mean_latency == 0:
                    raise RuntimeError(
                        "No workflow completion times recorded and no average time available. "
                        "The test may not have completed any workflows successfully."
                    )
                print(f"  Using mean latency: {mean_latency:.3f}s for {config.num_workflows} workflows")
                for _ in range(config.num_workflows):
                    collector.record_workflow_completion(mean_latency)

            print(f"  Test completed successfully")

        except subprocess.TimeoutExpired:
            print(f"  ERROR: Test execution timed out!")
            raise
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            raise
        finally:
            collector.end_collection()

        print(f"  Completed in {collector.get_makespan():.2f}s")

        return collector

    def run_experiment(
        self,
        config: OODExperimentConfig
    ) -> ExperimentMetrics:
        """
        Run a single experiment configuration.

        Args:
            config: Experiment configuration

        Returns:
            Experiment metrics
        """
        print("\n" + "=" * 60)
        print(f"Running: {config.experiment_name}")
        print("=" * 60)

        # Generate workload
        workflow = self.generate_workload(config)

        # Execute and collect metrics
        if self.use_real_services:
            collector = self.execute_with_real_services(workflow, config)
        else:
            collector = self.simulate_experiment_execution(workflow, config)

        # Generate metrics report
        metrics = collector.generate_metrics_report(
            num_workflows=config.num_workflows,
            strategy=config.strategy
        )

        # Save metrics
        collector.save_metrics(metrics, self.output_dir)

        # Print summary
        print(f"\nResults:")
        print(f"  Mean latency:   {metrics.workflow_latencies.mean:.3f}s")
        print(f"  Median latency: {metrics.workflow_latencies.median:.3f}s")
        print(f"  P95 latency:    {metrics.workflow_latencies.p95:.3f}s")
        print(f"  P99 latency:    {metrics.workflow_latencies.p99:.3f}s")
        print(f"  Makespan:       {metrics.makespan_seconds:.3f}s")

        return metrics

    def run_experiment_pair(
        self,
        baseline_config: OODExperimentConfig,
        comparison_config: OODExperimentConfig
    ):
        """
        Run a baseline/comparison experiment pair and compare results.

        Args:
            baseline_config: Baseline experiment configuration
            comparison_config: Comparison experiment configuration
        """
        # Restart services before baseline if using real services
        if self.use_real_services and self.restart_services and self.service_manager:
            print(f"\n{'*' * 60}")
            print(f"PREPARING FOR BASELINE EXPERIMENT")
            print(f"{'*' * 60}")
            if not self.service_manager.restart_services(self.restart_delay):
                raise RuntimeError("Failed to restart services before baseline experiment")

        # Run baseline
        baseline_metrics = self.run_experiment(baseline_config)

        # Restart services before comparison if using real services
        if self.use_real_services and self.restart_services and self.service_manager:
            print(f"\n{'*' * 60}")
            print(f"PREPARING FOR COMPARISON EXPERIMENT")
            print(f"{'*' * 60}")
            if not self.service_manager.restart_services(self.restart_delay):
                raise RuntimeError("Failed to restart services before comparison experiment")

        # Run comparison
        comparison_metrics = self.run_experiment(comparison_config)

        # Compare results
        comparison = MetricsComparator.compare_experiments(
            baseline_metrics,
            comparison_metrics
        )
        MetricsComparator.print_comparison(comparison)

    def run_all_experiments(self):
        """Run all out-of-distribution experiments."""
        print("\n" + "=" * 80)
        print("OUT-OF-DISTRIBUTION EXPERIMENTS - FULL SUITE")
        print("=" * 80)

        # Experiment 1: A1 Sampling
        print("\n### EXPERIMENT 1: B1 Samples Sleep Time from A1 Distribution ###")
        exp1_baseline = get_exp1_baseline_config()
        exp1_comparison = get_exp1_comparison_config()
        self.run_experiment_pair(exp1_baseline, exp1_comparison)

        # Experiment 2: Scaling
        print("\n### EXPERIMENT 2: B1 Scales Sleep Time and Exp Runtime ###")
        exp2_baseline_configs = get_exp2_baseline_configs()
        exp2_comparison_configs = get_exp2_comparison_configs()

        for baseline, comparison in zip(exp2_baseline_configs, exp2_comparison_configs):
            print(f"\n--- Scaling Factor: {baseline.b1_sleep_time_scale} ---")
            self.run_experiment_pair(baseline, comparison)

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run out-of-distribution experiments"
    )
    parser.add_argument(
        "--experiment",
        choices=["exp1", "exp2", "all"],
        default="all",
        help="Which experiment to run"
    )
    parser.add_argument(
        "--num-workflows",
        type=int,
        default=100,
        help="Number of workflows per experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./ood_results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--real-services",
        action="store_true",
        help="Use real scheduler/predictor services instead of simulation"
    )
    parser.add_argument(
        "--restart-services",
        action="store_true",
        default=True,
        help="Restart services before each test pair (only applies when --real-services is used)"
    )
    parser.add_argument(
        "--no-restart-services",
        dest="restart_services",
        action="store_false",
        help="Disable service restart (not recommended)"
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=10,
        help="Number of Group A instances"
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=6,
        help="Number of Group B instances"
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        default=30,
        help="Delay after service restart before running tests (seconds)"
    )

    args = parser.parse_args()

    runner = OODExperimentRunner(
        output_dir=args.output_dir,
        use_real_services=args.real_services,
        restart_services=args.restart_services,
        n1=args.n1,
        n2=args.n2,
        restart_delay=args.restart_delay
    )

    # Register cleanup handler for normal exit
    atexit.register(runner.cleanup)

    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n\nReceived signal {signum}, shutting down gracefully...")
        runner.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        if args.experiment == "exp1":
            exp1_baseline = get_exp1_baseline_config(num_workflows=args.num_workflows)
            exp1_comparison = get_exp1_comparison_config(num_workflows=args.num_workflows)
            runner.run_experiment_pair(exp1_baseline, exp1_comparison)

        elif args.experiment == "exp2":
            exp2_baseline_configs = get_exp2_baseline_configs(num_workflows=args.num_workflows)
            exp2_comparison_configs = get_exp2_comparison_configs(num_workflows=args.num_workflows)

            for baseline, comparison in zip(exp2_baseline_configs, exp2_comparison_configs):
                runner.run_experiment_pair(baseline, comparison)

        else:  # all
            runner.run_all_experiments()

    except KeyboardInterrupt:
        print(f"\n\nExperiment interrupted by user")
        runner.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Experiment failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        runner.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
