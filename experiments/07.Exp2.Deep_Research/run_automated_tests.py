#!/usr/bin/env python3
"""
Automated Testing Script for Experiment 07: Multi-Model Workflow with B1/B2 Split and Merge

This script automates:
1. Service management (stop/start with different instance configurations)
2. Test execution across parameter matrix
3. Result collection and aggregation
4. Report generation with visualizations

Usage:
    python run_automated_tests.py                           # Run full test suite
    python run_automated_tests.py --quick                   # Quick test mode
    python run_automated_tests.py --config custom.yaml      # Custom config
    python run_automated_tests.py --report-only             # Generate reports only
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import traceback

# Third-party imports (for visualization)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration management for automated testing."""

    def __init__(self, config_file: str = "test_config.yaml", quick_mode: bool = False):
        """Load configuration from YAML file."""
        self.config_file = config_file
        self.quick_mode = quick_mode

        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # Use quick test config if quick mode is enabled
        if quick_mode and 'quick_test' in self.config:
            self.config['test_matrix'] = self.config['quick_test']

    def get_test_matrix(self) -> Dict:
        """Get test matrix configuration."""
        return self.config.get('test_matrix', {})

    def get_execution_config(self) -> Dict:
        """Get execution configuration."""
        return self.config.get('execution', {})

    def get_reporting_config(self) -> Dict:
        """Get reporting configuration."""
        return self.config.get('reporting', {})

    def get_instance_configs(self) -> List[Dict]:
        """Get list of instance configurations to test."""
        return self.config['test_matrix'].get('instance_configs', [])


# ============================================================================
# Service Manager
# ============================================================================

class ServiceManager:
    """Manages service lifecycle (start, stop, health checks)."""

    def __init__(self, script_dir: Path, logger: logging.Logger):
        """
        Initialize ServiceManager.

        Args:
            script_dir: Directory containing service scripts
            logger: Logger instance
        """
        self.script_dir = script_dir
        self.logger = logger
        self.start_script = script_dir / "start_all_services.sh"
        self.stop_script = script_dir / "stop_all_services.sh"

        # Verify scripts exist
        if not self.start_script.exists():
            raise FileNotFoundError(f"Start script not found: {self.start_script}")
        if not self.stop_script.exists():
            raise FileNotFoundError(f"Stop script not found: {self.stop_script}")

    def stop_services(self) -> bool:
        """
        Stop all running services.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Stopping all services...")

        try:
            result = subprocess.run(
                [str(self.stop_script)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                self.logger.info("Services stopped successfully")
                return True
            else:
                self.logger.error(f"Failed to stop services: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Service stop timed out after 120 seconds")
            return False
        except Exception as e:
            self.logger.error(f"Error stopping services: {e}")
            return False

    def start_services(self, n1: int, n2: int, rebuild_docker: bool = False) -> bool:
        """
        Start services with specified instance configuration.

        Args:
            n1: Number of instances in Group A
            n2: Number of instances in Group B
            rebuild_docker: Whether to rebuild Docker images

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Starting services with N1={n1}, N2={n2}")

        try:
            # Prepare environment variables
            env = os.environ.copy()
            env['N1'] = str(n1)
            env['N2'] = str(n2)

            # Add rebuild flag if requested
            if rebuild_docker:
                env['REBUILD'] = '1'

            result = subprocess.run(
                [str(self.start_script)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode == 0:
                self.logger.info("Services started successfully")
                self.logger.debug(f"Start output: {result.stdout}")
                return True
            else:
                self.logger.error(f"Failed to start services: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Service start timed out after 300 seconds")
            return False
        except Exception as e:
            self.logger.error(f"Error starting services: {e}")
            return False

    def health_check(self, timeout: int = 60) -> bool:
        """
        Perform health check on all services.

        Args:
            timeout: Maximum time to wait for services to be healthy

        Returns:
            True if all services are healthy, False otherwise
        """
        self.logger.info("Performing health check...")

        import requests

        # Services to check
        services = [
            ("Predictor", "http://localhost:8099/health"),
            ("Scheduler A", "http://localhost:8100/health"),
            ("Scheduler B", "http://localhost:8200/health"),
        ]

        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = True

            for service_name, url in services:
                try:
                    response = requests.get(url, timeout=2)
                    if response.status_code != 200:
                        self.logger.warning(f"{service_name} not healthy yet")
                        all_healthy = False
                except requests.exceptions.RequestException:
                    self.logger.warning(f"{service_name} not reachable yet")
                    all_healthy = False

            if all_healthy:
                self.logger.info("All services are healthy")
                return True

            time.sleep(5)

        self.logger.error(f"Health check failed after {timeout} seconds")
        return False

    def restart_services(self, n1: int, n2: int, rebuild_docker: bool = False,
                         restart_delay: int = 30) -> bool:
        """
        Restart services with new configuration.

        Args:
            n1: Number of instances in Group A
            n2: Number of instances in Group B
            rebuild_docker: Whether to rebuild Docker images
            restart_delay: Delay after restart before proceeding

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Restarting services with N1={n1}, N2={n2}")

        # Stop existing services
        if not self.stop_services():
            self.logger.warning("Failed to stop services, continuing anyway...")

        # Wait a bit for cleanup
        time.sleep(5)

        # Start services with new configuration
        if not self.start_services(n1, n2, rebuild_docker):
            return False

        # Wait for services to stabilize
        self.logger.info(f"Waiting {restart_delay} seconds for services to stabilize...")
        time.sleep(restart_delay)

        # Health check
        if not self.health_check():
            return False

        return True


# ============================================================================
# Test Runner
# ============================================================================

class TestRunner:
    """Executes test scenarios with different parameters."""

    def __init__(self, script_dir: Path, logger: logging.Logger):
        """
        Initialize TestRunner.

        Args:
            script_dir: Directory containing test scripts
            logger: Logger instance
        """
        self.script_dir = script_dir
        self.logger = logger
        self.test_script = script_dir / "test_dynamic_workflow.py"

        if not self.test_script.exists():
            raise FileNotFoundError(f"Test script not found: {self.test_script}")

    def run_test(self, config: Dict, timeout: int = 600) -> Optional[Dict]:
        """
        Run a single test with specified configuration.

        Args:
            config: Test configuration with keys:
                - num_workflows: int
                - qps: float
                - gqps: float (optional, global QPS limit)
                - strategies: List[str]
                - warmup: float
                - seed: int
            timeout: Maximum time to wait for test completion (seconds)

        Returns:
            Dict with test results, or None if test failed
        """
        # Build command
        cmd = [
            "uv", "run", "python3", str(self.test_script),
            "--num-workflows", str(config['num_workflows']),
            "--qps", str(config['qps']),
            "--seed", str(config['seed']),
        ]

        # Add strategies
        for strategy in config['strategies']:
            cmd.extend(["--strategies", strategy])

        # Add gqps if specified
        if config.get('gqps') is not None:
            cmd.extend(["--gqps", str(config['gqps'])])

        # Add warmup if specified
        if config.get('warmup', 0.0) > 0:
            cmd.extend(["--warmup", str(config['warmup'])])

        self.logger.info(f"Running test: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0:
                self.logger.info("Test completed successfully")
                self.logger.debug(f"Test stdout: {result.stdout[:200]}...")
                self.logger.debug(f"Test stderr: {result.stderr[:200]}...")

                # Extract result file path from output
                # Try stdout first, then stderr (logging often goes to stderr)
                result_file = self._extract_result_file(result.stdout)
                if not result_file:
                    self.logger.debug("Result file not found in stdout, trying stderr...")
                    result_file = self._extract_result_file(result.stderr)

                if result_file:
                    return {
                        'success': True,
                        'result_file': result_file,
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
                else:
                    self.logger.error("Could not find result file in test output (stdout or stderr)")
                    self.logger.debug(f"Full stdout:\n{result.stdout}")
                    self.logger.debug(f"Full stderr:\n{result.stderr}")
                    return {
                        'success': False,
                        'error': 'Result file not found',
                        'stdout': result.stdout,
                        'stderr': result.stderr
                    }
            else:
                self.logger.error(f"Test failed with return code {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }

        except subprocess.TimeoutExpired:
            self.logger.error(f"Test timed out after {timeout} seconds")
            return {
                'success': False,
                'error': f'Timeout after {timeout} seconds'
            }
        except Exception as e:
            self.logger.error(f"Error running test: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _extract_result_file(self, output: str) -> Optional[str]:
        """
        Extract result file path from test output.

        Args:
            output: Test stdout

        Returns:
            Path to result file, or None if not found
        """
        import re

        # Look for pattern: "Results saved to: <path>"
        # Pattern 1: "Results saved to: path/to/file.json"
        pattern1 = r'Results saved to:\s+(.+?\.json)'
        match = re.search(pattern1, output)
        if match:
            result_file = match.group(1).strip()
            result_path = Path(result_file)
            if not result_path.is_absolute():
                result_path = self.script_dir / result_path
            if result_path.exists():
                self.logger.debug(f"Found result file (pattern 1): {result_path}")
                return str(result_path)
            else:
                self.logger.warning(f"Result file not found at: {result_path}")

        # Pattern 2: Look for any line containing "results_workflow_b1b2_*.json"
        pattern2 = r'(results/results_workflow_b1b2_\d+\.json)'
        match = re.search(pattern2, output)
        if match:
            result_file = match.group(1).strip()
            result_path = Path(result_file)
            if not result_path.is_absolute():
                result_path = self.script_dir / result_path
            if result_path.exists():
                self.logger.debug(f"Found result file (pattern 2): {result_path}")
                return str(result_path)
            else:
                self.logger.warning(f"Result file not found at: {result_path}")

        # Pattern 3: Look for just the filename
        pattern3 = r'(results_workflow_b1b2_\d+\.json)'
        matches = re.findall(pattern3, output)
        if matches:
            for result_file in matches:
                # Try in results/ subdirectory
                result_path = self.script_dir / "results" / result_file
                if result_path.exists():
                    self.logger.debug(f"Found result file (pattern 3): {result_path}")
                    return str(result_path)
                # Try in current directory
                result_path = self.script_dir / result_file
                if result_path.exists():
                    self.logger.debug(f"Found result file (pattern 3, current dir): {result_path}")
                    return str(result_path)

        # If still not found, list all JSON files in results directory
        results_dir = self.script_dir / "results"
        if results_dir.exists():
            json_files = sorted(results_dir.glob("results_workflow_b1b2_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if json_files:
                latest_file = json_files[0]
                self.logger.warning(f"Could not extract from output, using latest result file: {latest_file}")
                return str(latest_file)
            else:
                self.logger.error(f"results directory exists but no result files found in: {results_dir}")
        else:
            self.logger.error(f"results directory does not exist: {results_dir}")

        # Log relevant portions of the output for debugging
        output_lines = output.split('\n')
        relevant_lines = [line for line in output_lines if 'result' in line.lower() or 'saved' in line.lower()]
        if relevant_lines:
            self.logger.error(f"Lines containing 'result' or 'saved':\n" + "\n".join(relevant_lines[:10]))
        else:
            self.logger.error(f"No lines containing 'result' or 'saved' found in output")

        self.logger.debug(f"Full output length: {len(output)} characters")
        self.logger.debug(f"Output snippet (first 1000 chars):\n{output[:1000]}")

        return None


# ============================================================================
# Result Collector
# ============================================================================

class ResultCollector:
    """Collects and aggregates test results."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize ResultCollector.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    def parse_result_file(self, result_file: str) -> Optional[Dict]:
        """
        Parse a JSON result file.

        Args:
            result_file: Path to result file

        Returns:
            Parsed result dict, or None if parsing failed
        """
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            self.logger.debug(f"Parsed result file: {result_file}")
            return data

        except Exception as e:
            self.logger.error(f"Error parsing result file {result_file}: {e}")
            return None

    def aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate multiple test results.

        Args:
            results: List of test result dicts, each with keys:
                - config: Test configuration
                - result_data: Parsed result data
                - instance_config: Instance configuration (N1, N2)

        Returns:
            Aggregated results dict
        """
        aggregated = {
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r.get('success', False)),
            'failed_tests': sum(1 for r in results if not r.get('success', False)),
            'results_by_instance_config': {},
            'results_by_strategy': {},
            'results_by_qps': {},
            'results_by_workflows': {},
            'all_results': results
        }

        # Group results by different dimensions
        for result in results:
            if not result.get('success', False):
                continue

            config = result.get('config', {})
            instance_config = result.get('instance_config', {})
            result_data = result.get('result_data', {})

            # By instance config
            instance_key = f"N1={instance_config.get('N1')}_N2={instance_config.get('N2')}"
            if instance_key not in aggregated['results_by_instance_config']:
                aggregated['results_by_instance_config'][instance_key] = []
            aggregated['results_by_instance_config'][instance_key].append(result)

            # By strategy
            for strategy in config.get('strategies', []):
                if strategy not in aggregated['results_by_strategy']:
                    aggregated['results_by_strategy'][strategy] = []
                aggregated['results_by_strategy'][strategy].append(result)

            # By QPS
            qps = config.get('qps')
            if qps not in aggregated['results_by_qps']:
                aggregated['results_by_qps'][qps] = []
            aggregated['results_by_qps'][qps].append(result)

            # By workflows
            workflows = config.get('num_workflows')
            if workflows not in aggregated['results_by_workflows']:
                aggregated['results_by_workflows'][workflows] = []
            aggregated['results_by_workflows'][workflows].append(result)

        return aggregated


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generates reports and visualizations."""

    def __init__(self, output_dir: Path, logger: logging.Logger):
        """
        Initialize ReportGenerator.

        Args:
            output_dir: Directory to save reports
            logger: Logger instance
        """
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, aggregated_results: Dict, config: Config) -> str:
        """
        Generate HTML report.

        Args:
            aggregated_results: Aggregated test results
            config: Test configuration

        Returns:
            Path to generated HTML file
        """
        self.logger.info("Generating HTML report...")

        html_file = self.output_dir / "summary_report.html"

        # Generate HTML content
        html_content = self._create_html_report(aggregated_results, config)

        # Write to file
        with open(html_file, 'w') as f:
            f.write(html_content)

        self.logger.info(f"HTML report generated: {html_file}")
        return str(html_file)

    def _create_html_report(self, aggregated_results: Dict, config: Config) -> str:
        """Create HTML report content."""

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Automated Test Report - Experiment 07</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1976d2;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        .success {{
            color: green;
        }}
        .failure {{
            color: red;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <h1>Automated Test Report: Experiment 07</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>Test Summary</h2>
        <div class="metric">
            <div class="metric-value">{aggregated_results['total_tests']}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric">
            <div class="metric-value success">{aggregated_results['successful_tests']}</div>
            <div class="metric-label">Successful</div>
        </div>
        <div class="metric">
            <div class="metric-value failure">{aggregated_results['failed_tests']}</div>
            <div class="metric-label">Failed</div>
        </div>
    </div>

    <div class="summary">
        <h2>Instance Configuration Comparison</h2>
        {self._create_instance_comparison_table(aggregated_results)}
    </div>

    <div class="summary">
        <h2>Strategy Performance Comparison</h2>
        {self._create_strategy_comparison_table(aggregated_results)}
    </div>

    <div class="summary">
        <h2>QPS Impact Analysis</h2>
        {self._create_qps_analysis_table(aggregated_results)}
    </div>

    <div class="summary">
        <h2>Detailed Results</h2>
        {self._create_detailed_results_table(aggregated_results)}
    </div>
</body>
</html>"""

        return html

    def _create_instance_comparison_table(self, aggregated_results: Dict) -> str:
        """Create instance configuration comparison table."""

        results_by_config = aggregated_results.get('results_by_instance_config', {})

        if not results_by_config:
            return "<p>No results available</p>"

        html = """<table>
            <tr>
                <th>Instance Config</th>
                <th>Total Tests</th>
                <th>Avg Workflow Time (s)</th>
                <th>P95 Workflow Time (s)</th>
                <th>Avg A Completion (s)</th>
                <th>Avg B1 Completion (s)</th>
                <th>Avg B2 Completion (s)</th>
            </tr>"""

        for config_key, results in sorted(results_by_config.items()):
            # Calculate aggregate metrics
            workflow_times = []
            a_times = []
            b1_times = []
            b2_times = []

            for result in results:
                result_data = result.get('result_data', {})
                for strategy_result in result_data.get('results', []):
                    strategy_name = strategy_result.get('strategy')
                    workflows = strategy_result.get('workflows', {})

                    wf_time = workflows.get('avg_workflow_time')
                    if wf_time is not None:
                        workflow_times.append(wf_time)

                    a_tasks = strategy_result.get('a_tasks', {})
                    a_avg = a_tasks.get('avg_completion_time')
                    if a_avg is not None:
                        a_times.append(a_avg)

                    b1_tasks = strategy_result.get('b1_tasks', {})
                    b1_avg = b1_tasks.get('avg_completion_time')
                    if b1_avg is not None:
                        b1_times.append(b1_avg)

                    b2_tasks = strategy_result.get('b2_tasks', {})
                    b2_avg = b2_tasks.get('avg_completion_time')
                    if b2_avg is not None:
                        b2_times.append(b2_avg)

            # Calculate statistics
            avg_workflow = np.mean(workflow_times) if workflow_times else 0
            p95_workflow = np.percentile(workflow_times, 95) if workflow_times else 0
            avg_a = np.mean(a_times) if a_times else 0
            avg_b1 = np.mean(b1_times) if b1_times else 0
            avg_b2 = np.mean(b2_times) if b2_times else 0

            html += f"""<tr>
                <td><strong>{config_key}</strong></td>
                <td>{len(results)}</td>
                <td>{avg_workflow:.2f}</td>
                <td>{p95_workflow:.2f}</td>
                <td>{avg_a:.2f}</td>
                <td>{avg_b1:.2f}</td>
                <td>{avg_b2:.2f}</td>
            </tr>"""

        html += "</table>"
        return html

    def _create_strategy_comparison_table(self, aggregated_results: Dict) -> str:
        """Create strategy comparison table."""

        results_by_strategy = aggregated_results.get('results_by_strategy', {})

        if not results_by_strategy:
            return "<p>No results available</p>"

        html = """<table>
            <tr>
                <th>Strategy</th>
                <th>Total Tests</th>
                <th>Avg Workflow Time (s)</th>
                <th>P95 Workflow Time (s)</th>
                <th>P99 Workflow Time (s)</th>
            </tr>"""

        for strategy, results in sorted(results_by_strategy.items()):
            # Calculate aggregate metrics
            workflow_times = []

            for result in results:
                result_data = result.get('result_data', {})
                strategy_data = result_data.get('results', {}).get(strategy, {})
                workflow_metrics = strategy_data.get('workflow_metrics', {})

                wf_time = workflow_metrics.get('avg_workflow_time')
                if wf_time is not None:
                    workflow_times.append(wf_time)

            # Calculate statistics
            avg_workflow = np.mean(workflow_times) if workflow_times else 0
            p95_workflow = np.percentile(workflow_times, 95) if workflow_times else 0
            p99_workflow = np.percentile(workflow_times, 99) if workflow_times else 0

            html += f"""<tr>
                <td><strong>{strategy}</strong></td>
                <td>{len(results)}</td>
                <td>{avg_workflow:.2f}</td>
                <td>{p95_workflow:.2f}</td>
                <td>{p99_workflow:.2f}</td>
            </tr>"""

        html += "</table>"
        return html

    def _create_qps_analysis_table(self, aggregated_results: Dict) -> str:
        """Create QPS analysis table."""

        results_by_qps = aggregated_results.get('results_by_qps', {})

        if not results_by_qps:
            return "<p>No results available</p>"

        html = """<table>
            <tr>
                <th>QPS</th>
                <th>Total Tests</th>
                <th>Avg Workflow Time (s)</th>
                <th>P95 Workflow Time (s)</th>
            </tr>"""

        for qps, results in sorted(results_by_qps.items()):
            # Calculate aggregate metrics
            workflow_times = []

            for result in results:
                result_data = result.get('result_data', {})
                for strategy_result in result_data.get('results', []):
                    workflows = strategy_result.get('workflows', {})
                    wf_time = workflows.get('avg_workflow_time')
                    if wf_time is not None:
                        workflow_times.append(wf_time)

            # Calculate statistics
            avg_workflow = np.mean(workflow_times) if workflow_times else 0
            p95_workflow = np.percentile(workflow_times, 95) if workflow_times else 0

            html += f"""<tr>
                <td><strong>{qps}</strong></td>
                <td>{len(results)}</td>
                <td>{avg_workflow:.2f}</td>
                <td>{p95_workflow:.2f}</td>
            </tr>"""

        html += "</table>"
        return html

    def _create_detailed_results_table(self, aggregated_results: Dict) -> str:
        """Create detailed results table."""

        all_results = aggregated_results.get('all_results', [])

        if not all_results:
            return "<p>No results available</p>"

        html = """<table>
            <tr>
                <th>Instance Config</th>
                <th>Workflows</th>
                <th>QPS</th>
                <th>GQPS</th>
                <th>Strategy</th>
                <th>Warmup</th>
                <th>Workflow Time (s)</th>
                <th>Status</th>
            </tr>"""

        for result in all_results:
            config = result.get('config', {})
            instance_config = result.get('instance_config', {})
            result_data = result.get('result_data', {})

            instance_key = f"N1={instance_config.get('N1')}_N2={instance_config.get('N2')}"
            workflows = config.get('num_workflows', 'N/A')
            qps = config.get('qps', 'N/A')
            gqps = config.get('gqps', '-')
            if gqps is None:
                gqps = '-'
            warmup = config.get('warmup', 0.0)
            success = result.get('success', False)

            # Get metrics for each strategy
            for strategy_result in result_data.get('results', []):
                strategy_name = strategy_result.get('strategy')
                workflows_data = strategy_result.get('workflows', {})
                wf_time = workflows_data.get('avg_workflow_time', 'N/A')
                if isinstance(wf_time, float):
                    wf_time = f"{wf_time:.2f}"

                status_class = 'success' if success else 'failure'
                status_text = 'SUCCESS' if success else 'FAILED'

                html += f"""<tr>
                    <td>{instance_key}</td>
                    <td>{workflows}</td>
                    <td>{qps}</td>
                    <td>{gqps}</td>
                    <td>{strategy_name}</td>
                    <td>{warmup}</td>
                    <td>{wf_time}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>"""

        html += "</table>"
        return html

    def generate_markdown_report(self, aggregated_results: Dict, config: Config) -> str:
        """
        Generate Markdown report.

        Args:
            aggregated_results: Aggregated test results
            config: Test configuration

        Returns:
            Path to generated Markdown file
        """
        self.logger.info("Generating Markdown report...")

        md_file = self.output_dir / "summary_report.md"

        # Generate Markdown content
        md_content = self._create_markdown_report(aggregated_results, config)

        # Write to file
        with open(md_file, 'w') as f:
            f.write(md_content)

        self.logger.info(f"Markdown report generated: {md_file}")
        return str(md_file)

    def _create_markdown_report(self, aggregated_results: Dict, config: Config) -> str:
        """Create Markdown report content."""

        md = f"""# Automated Test Report: Experiment 07

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Test Summary

- **Total Tests:** {aggregated_results['total_tests']}
- **Successful:** {aggregated_results['successful_tests']}
- **Failed:** {aggregated_results['failed_tests']}

## Instance Configuration Comparison

{self._create_instance_comparison_markdown(aggregated_results)}

## Strategy Performance Comparison

{self._create_strategy_comparison_markdown(aggregated_results)}

## QPS Impact Analysis

{self._create_qps_analysis_markdown(aggregated_results)}

## Detailed Results

{self._create_detailed_results_markdown(aggregated_results)}

---
*Report generated by automated testing script*
"""

        return md

    def _create_instance_comparison_markdown(self, aggregated_results: Dict) -> str:
        """Create instance configuration comparison in Markdown."""

        results_by_config = aggregated_results.get('results_by_instance_config', {})

        if not results_by_config:
            return "No results available"

        md = """| Instance Config | Total Tests | Avg Workflow Time (s) | P95 Workflow Time (s) | Avg A Completion (s) | Avg B1 Completion (s) | Avg B2 Completion (s) |
|----------------|-------------|----------------------|----------------------|---------------------|----------------------|----------------------|
"""

        for config_key, results in sorted(results_by_config.items()):
            # Calculate aggregate metrics
            workflow_times = []
            a_times = []
            b1_times = []
            b2_times = []

            for result in results:
                result_data = result.get('result_data', {})
                for strategy_result in result_data.get('results', []):
                    strategy_name = strategy_result.get('strategy')
                    workflows = strategy_result.get('workflows', {})

                    wf_time = workflows.get('avg_workflow_time')
                    if wf_time is not None:
                        workflow_times.append(wf_time)

                    a_tasks = strategy_result.get('a_tasks', {})
                    a_avg = a_tasks.get('avg_completion_time')
                    if a_avg is not None:
                        a_times.append(a_avg)

                    b1_tasks = strategy_result.get('b1_tasks', {})
                    b1_avg = b1_tasks.get('avg_completion_time')
                    if b1_avg is not None:
                        b1_times.append(b1_avg)

                    b2_tasks = strategy_result.get('b2_tasks', {})
                    b2_avg = b2_tasks.get('avg_completion_time')
                    if b2_avg is not None:
                        b2_times.append(b2_avg)

            # Calculate statistics
            avg_workflow = np.mean(workflow_times) if workflow_times else 0
            p95_workflow = np.percentile(workflow_times, 95) if workflow_times else 0
            avg_a = np.mean(a_times) if a_times else 0
            avg_b1 = np.mean(b1_times) if b1_times else 0
            avg_b2 = np.mean(b2_times) if b2_times else 0

            md += f"| **{config_key}** | {len(results)} | {avg_workflow:.2f} | {p95_workflow:.2f} | {avg_a:.2f} | {avg_b1:.2f} | {avg_b2:.2f} |\n"

        return md

    def _create_strategy_comparison_markdown(self, aggregated_results: Dict) -> str:
        """Create strategy comparison in Markdown."""

        results_by_strategy = aggregated_results.get('results_by_strategy', {})

        if not results_by_strategy:
            return "No results available"

        md = """| Strategy | Total Tests | Avg Workflow Time (s) | P95 Workflow Time (s) | P99 Workflow Time (s) |
|----------|-------------|----------------------|----------------------|----------------------|
"""

        for strategy, results in sorted(results_by_strategy.items()):
            # Calculate aggregate metrics
            workflow_times = []

            for result in results:
                result_data = result.get('result_data', {})
                strategy_data = result_data.get('results', {}).get(strategy, {})
                workflow_metrics = strategy_data.get('workflow_metrics', {})

                wf_time = workflow_metrics.get('avg_workflow_time')
                if wf_time is not None:
                    workflow_times.append(wf_time)

            # Calculate statistics
            avg_workflow = np.mean(workflow_times) if workflow_times else 0
            p95_workflow = np.percentile(workflow_times, 95) if workflow_times else 0
            p99_workflow = np.percentile(workflow_times, 99) if workflow_times else 0

            md += f"| **{strategy}** | {len(results)} | {avg_workflow:.2f} | {p95_workflow:.2f} | {p99_workflow:.2f} |\n"

        return md

    def _create_qps_analysis_markdown(self, aggregated_results: Dict) -> str:
        """Create QPS analysis in Markdown."""

        results_by_qps = aggregated_results.get('results_by_qps', {})

        if not results_by_qps:
            return "No results available"

        md = """| QPS | Total Tests | Avg Workflow Time (s) | P95 Workflow Time (s) |
|-----|-------------|----------------------|----------------------|
"""

        for qps, results in sorted(results_by_qps.items()):
            # Calculate aggregate metrics
            workflow_times = []

            for result in results:
                result_data = result.get('result_data', {})
                for strategy_result in result_data.get('results', []):
                    workflows = strategy_result.get('workflows', {})
                    wf_time = workflows.get('avg_workflow_time')
                    if wf_time is not None:
                        workflow_times.append(wf_time)

            # Calculate statistics
            avg_workflow = np.mean(workflow_times) if workflow_times else 0
            p95_workflow = np.percentile(workflow_times, 95) if workflow_times else 0

            md += f"| **{qps}** | {len(results)} | {avg_workflow:.2f} | {p95_workflow:.2f} |\n"

        return md

    def _create_detailed_results_markdown(self, aggregated_results: Dict) -> str:
        """Create detailed results in Markdown."""

        all_results = aggregated_results.get('all_results', [])

        if not all_results:
            return "No results available"

        md = """| Instance Config | Workflows | QPS | GQPS | Strategy | Warmup | Workflow Time (s) | Status |
|----------------|-----------|-----|------|----------|--------|------------------|--------|
"""

        for result in all_results:
            config = result.get('config', {})
            instance_config = result.get('instance_config', {})
            result_data = result.get('result_data', {})

            instance_key = f"N1={instance_config.get('N1')}_N2={instance_config.get('N2')}"
            workflows = config.get('num_workflows', 'N/A')
            qps = config.get('qps', 'N/A')
            gqps = config.get('gqps', '-')
            if gqps is None:
                gqps = '-'
            warmup = config.get('warmup', 0.0)
            success = result.get('success', False)

            # Get metrics for each strategy
            for strategy_result in result_data.get('results', []):
                strategy_name = strategy_result.get('strategy')
                workflows_data = strategy_result.get('workflows', {})
                wf_time = workflows_data.get('avg_workflow_time', 'N/A')
                if isinstance(wf_time, float):
                    wf_time = f"{wf_time:.2f}"

                status_text = 'SUCCESS' if success else 'FAILED'

                md += f"| {instance_key} | {workflows} | {qps} | {gqps} | {strategy_name} | {warmup} | {wf_time} | {status_text} |\n"

        return md

    def generate_plots(self, aggregated_results: Dict) -> List[str]:
        """
        Generate visualization plots.

        Args:
            aggregated_results: Aggregated test results

        Returns:
            List of generated plot file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping plot generation")
            return []

        self.logger.info("Generating plots...")

        plot_files = []

        # Instance configuration comparison plot
        plot_file = self._plot_instance_comparison(aggregated_results)
        if plot_file:
            plot_files.append(plot_file)

        # Strategy comparison plot
        plot_file = self._plot_strategy_comparison(aggregated_results)
        if plot_file:
            plot_files.append(plot_file)

        # QPS impact plot
        plot_file = self._plot_qps_impact(aggregated_results)
        if plot_file:
            plot_files.append(plot_file)

        self.logger.info(f"Generated {len(plot_files)} plots")
        return plot_files

    def _plot_instance_comparison(self, aggregated_results: Dict) -> Optional[str]:
        """Generate instance configuration comparison plot."""

        try:
            results_by_config = aggregated_results.get('results_by_instance_config', {})

            if not results_by_config:
                return None

            # Prepare data
            configs = []
            avg_workflow_times = []
            p95_workflow_times = []

            for config_key, results in sorted(results_by_config.items()):
                workflow_times = []

                for result in results:
                    result_data = result.get('result_data', {})
                    for strategy_result in result_data.get('results', []):
                        workflows_data = strategy_result.get('workflows', {})
                        wf_time = workflows_data.get('avg_workflow_time')
                        if wf_time is not None:
                            workflow_times.append(wf_time)

                if workflow_times:
                    configs.append(config_key.replace('_', '\n'))
                    avg_workflow_times.append(np.mean(workflow_times))
                    p95_workflow_times.append(np.percentile(workflow_times, 95))

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(configs))
            width = 0.35

            ax.bar(x - width/2, avg_workflow_times, width, label='Avg Workflow Time')
            ax.bar(x + width/2, p95_workflow_times, width, label='P95 Workflow Time')

            ax.set_xlabel('Instance Configuration')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Workflow Time by Instance Configuration')
            ax.set_xticks(x)
            ax.set_xticklabels(configs)
            ax.legend()

            plt.tight_layout()

            plot_file = self.output_dir / "instance_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.debug(f"Generated instance comparison plot: {plot_file}")
            return str(plot_file)

        except Exception as e:
            self.logger.error(f"Error generating instance comparison plot: {e}")
            return None

    def _plot_strategy_comparison(self, aggregated_results: Dict) -> Optional[str]:
        """Generate strategy comparison plot."""

        try:
            results_by_strategy = aggregated_results.get('results_by_strategy', {})

            if not results_by_strategy:
                return None

            # Prepare data
            strategies = []
            avg_workflow_times = []
            p95_workflow_times = []
            p99_workflow_times = []

            for strategy, results in sorted(results_by_strategy.items()):
                workflow_times = []

                for result in results:
                    result_data = result.get('result_data', {})
                    strategy_data = result_data.get('results', {}).get(strategy, {})
                    workflow_metrics = strategy_data.get('workflow_metrics', {})

                    wf_time = workflow_metrics.get('avg_workflow_time')
                    if wf_time is not None:
                        workflow_times.append(wf_time)

                if workflow_times:
                    strategies.append(strategy)
                    avg_workflow_times.append(np.mean(workflow_times))
                    p95_workflow_times.append(np.percentile(workflow_times, 95))
                    p99_workflow_times.append(np.percentile(workflow_times, 99))

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(strategies))
            width = 0.25

            ax.bar(x - width, avg_workflow_times, width, label='Avg')
            ax.bar(x, p95_workflow_times, width, label='P95')
            ax.bar(x + width, p99_workflow_times, width, label='P99')

            ax.set_xlabel('Strategy')
            ax.set_ylabel('Workflow Time (seconds)')
            ax.set_title('Workflow Time by Strategy')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies)
            ax.legend()

            plt.tight_layout()

            plot_file = self.output_dir / "strategy_comparison.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.debug(f"Generated strategy comparison plot: {plot_file}")
            return str(plot_file)

        except Exception as e:
            self.logger.error(f"Error generating strategy comparison plot: {e}")
            return None

    def _plot_qps_impact(self, aggregated_results: Dict) -> Optional[str]:
        """Generate QPS impact plot."""

        try:
            results_by_qps = aggregated_results.get('results_by_qps', {})

            if not results_by_qps:
                return None

            # Prepare data
            qps_values = []
            avg_workflow_times = []
            p95_workflow_times = []

            for qps, results in sorted(results_by_qps.items()):
                workflow_times = []

                for result in results:
                    result_data = result.get('result_data', {})
                    for strategy_result in result_data.get('results', []):
                        workflows_data = strategy_result.get('workflows', {})
                        wf_time = workflows_data.get('avg_workflow_time')
                        if wf_time is not None:
                            workflow_times.append(wf_time)

                if workflow_times:
                    qps_values.append(qps)
                    avg_workflow_times.append(np.mean(workflow_times))
                    p95_workflow_times.append(np.percentile(workflow_times, 95))

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(qps_values, avg_workflow_times, marker='o', label='Avg Workflow Time')
            ax.plot(qps_values, p95_workflow_times, marker='s', label='P95 Workflow Time')

            ax.set_xlabel('QPS (Queries Per Second)')
            ax.set_ylabel('Workflow Time (seconds)')
            ax.set_title('Impact of QPS on Workflow Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_file = self.output_dir / "qps_impact.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.debug(f"Generated QPS impact plot: {plot_file}")
            return str(plot_file)

        except Exception as e:
            self.logger.error(f"Error generating QPS impact plot: {e}")
            return None


# ============================================================================
# Main Orchestrator
# ============================================================================

def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        output_dir: Directory to save log files
        verbose: Enable verbose logging

    Returns:
        Logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "automation.log"

    # Create logger
    logger = logging.getLogger("AutomatedTesting")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_progress(progress_file: Path, results: List[Dict], completed_tests: int,
                  total_tests: int, logger: logging.Logger):
    """
    Save test progress to file.

    Args:
        progress_file: Path to progress file
        results: List of test results
        completed_tests: Number of completed tests
        total_tests: Total number of tests
        logger: Logger instance
    """
    try:
        progress_data = {
            'completed_tests': completed_tests,
            'total_tests': total_tests,
            'last_updated': datetime.now().isoformat(),
            'results': results
        }

        # Write to temporary file first, then rename for atomicity
        temp_file = progress_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

        # Atomic rename
        temp_file.replace(progress_file)

        logger.debug(f"Progress saved: {completed_tests}/{total_tests} tests completed")

    except Exception as e:
        logger.error(f"Failed to save progress: {e}")


def run_test_matrix(config: Config, service_manager: ServiceManager,
                    test_runner: TestRunner, result_collector: ResultCollector,
                    logger: logging.Logger, output_dir: Path) -> List[Dict]:
    """
    Run the full test matrix.

    Args:
        config: Test configuration
        service_manager: Service manager instance
        test_runner: Test runner instance
        result_collector: Result collector instance
        logger: Logger instance
        output_dir: Output directory for saving progress

    Returns:
        List of test results
    """
    test_matrix = config.get_test_matrix()
    execution_config = config.get_execution_config()

    # Generate all test configurations
    test_configs = []

    # Get gqps values, default to [None] if not specified or empty
    gqps_values = test_matrix.get('gqps_values', [])
    if not gqps_values:
        gqps_values = [None]  # No global QPS limit

    for instance_config in test_matrix.get('instance_configs', []):
        for num_workflows in test_matrix.get('workflows', []):
            for qps in test_matrix.get('qps_values', []):
                for gqps in gqps_values:
                    for warmup in test_matrix.get('warmup_ratios', []):
                        test_configs.append({
                            'instance_config': instance_config,
                            'num_workflows': num_workflows,
                            'qps': qps,
                            'gqps': gqps,
                            'strategies': test_matrix.get('strategies', []),
                            'warmup': warmup,
                            'seed': execution_config.get('seed', 42)
                        })

    total_tests = len(test_configs)
    logger.info(f"Generated {total_tests} test configurations")

    # Progress tracking file
    progress_file = output_dir / "progress.json"

    # Load existing progress if resuming
    start_index = 0
    results = []
    if progress_file.exists():
        logger.info(f"Found existing progress file: {progress_file}")
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            results = progress_data.get('results', [])
            start_index = progress_data.get('completed_tests', 0)
            logger.info(f"Resuming from test {start_index + 1}/{total_tests} ({len(results)} results loaded)")
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}. Starting from beginning.")
            start_index = 0
            results = []

    # Run tests
    current_instance_config = None

    for i, test_config in enumerate(test_configs, 1):
        # Skip already completed tests when resuming
        if i <= start_index:
            logger.debug(f"Skipping test {i}/{total_tests} (already completed)")
            # Still need to track instance config for service management
            instance_config = test_config['instance_config']
            current_instance_config = instance_config
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Running test {i}/{total_tests}")
        logger.info(f"{'='*80}")

        instance_config = test_config['instance_config']

        # Restart services if instance configuration changed
        if current_instance_config != instance_config:
            logger.info(f"Instance configuration changed, restarting services...")
            n1 = instance_config['N1']
            n2 = instance_config['N2']

            success = service_manager.restart_services(
                n1=n1,
                n2=n2,
                rebuild_docker=execution_config.get('rebuild_docker', False),
                restart_delay=execution_config.get('service_restart_delay', 30)
            )

            if not success:
                logger.error(f"Failed to restart services for instance config: N1={n1}, N2={n2}")

                # Retry if configured
                if execution_config.get('retry_on_failure', False):
                    max_retries = execution_config.get('max_retries', 2)
                    for retry in range(1, max_retries + 1):
                        logger.info(f"Retry {retry}/{max_retries}...")
                        success = service_manager.restart_services(
                            n1=n1,
                            n2=n2,
                            rebuild_docker=execution_config.get('rebuild_docker', False),
                            restart_delay=execution_config.get('service_restart_delay', 30)
                        )
                        if success:
                            break

                if not success:
                    logger.error("Failed to restart services after retries, skipping tests for this config")
                    # Skip all tests for this instance config
                    continue

            current_instance_config = instance_config

        # Run test
        logger.info(f"Test config: {test_config}")

        test_result = test_runner.run_test(
            config=test_config,
            timeout=execution_config.get('timeout_per_test', 600)
        )

        # Process result
        if test_result and test_result.get('success', False):
            result_file = test_result.get('result_file')
            if result_file:
                result_data = result_collector.parse_result_file(result_file)
                if result_data:
                    results.append({
                        'success': True,
                        'config': test_config,
                        'instance_config': instance_config,
                        'result_file': result_file,
                        'result_data': result_data
                    })
                    logger.info(f"Test {i}/{total_tests} completed successfully")
                else:
                    logger.error(f"Failed to parse result file: {result_file}")
                    results.append({
                        'success': False,
                        'config': test_config,
                        'instance_config': instance_config,
                        'error': 'Failed to parse result file'
                    })
            else:
                logger.error("Test completed but no result file found")
                results.append({
                    'success': False,
                    'config': test_config,
                    'instance_config': instance_config,
                    'error': 'No result file'
                })
        else:
            error = test_result.get('error', 'Unknown error') if test_result else 'Test runner failed'
            logger.error(f"Test {i}/{total_tests} failed: {error}")
            results.append({
                'success': False,
                'config': test_config,
                'instance_config': instance_config,
                'error': error
            })

        # Save progress after each test
        save_progress(progress_file, results, i, total_tests, logger)
        logger.info(f"✓ Progress saved ({i}/{total_tests} tests completed)")

        # Wait between tests
        if i < total_tests:
            inter_test_delay = execution_config.get('inter_test_delay', 10)
            logger.info(f"Waiting {inter_test_delay} seconds before next test...")
            time.sleep(inter_test_delay)

    logger.info(f"\n{'='*80}")
    logger.info(f"Completed {len(results)}/{total_tests} tests")
    logger.info(f"{'='*80}\n")

    return results


def main():
    """Main entry point."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Automated testing script for Experiment 07",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full test suite
    python run_automated_tests.py

    # Quick test mode
    python run_automated_tests.py --quick

    # Custom configuration
    python run_automated_tests.py --config custom_config.yaml

    # Generate reports only (no testing)
    python run_automated_tests.py --report-only

    # Verbose logging
    python run_automated_tests.py --verbose
        """
    )

    parser.add_argument('--config', type=str, default='test_config.yaml',
                        help='Path to configuration file (default: test_config.yaml)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test mode with reduced parameters')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate reports only, skip testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--output-dir', type=str,
                        help='Override output directory for reports')

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.resolve()

    # Load configuration
    try:
        config = Config(config_file=args.config, quick_mode=args.quick)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Setup output directory
    reporting_config = config.get_reporting_config()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_base = Path(reporting_config.get('output_dir', './test_reports'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_base / f"run_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, verbose=args.verbose)

    logger.info("="*80)
    logger.info("Automated Testing Script for Experiment 07")
    logger.info("="*80)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Report only: {args.report_only}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    try:
        # Initialize managers
        service_manager = ServiceManager(script_dir, logger)
        test_runner = TestRunner(script_dir, logger)
        result_collector = ResultCollector(logger)
        report_generator = ReportGenerator(output_dir, logger)

        # Run tests or load existing results
        if args.report_only:
            logger.info("Report-only mode: skipping test execution")
            # TODO: Load results from previous run
            logger.error("Report-only mode not yet implemented")
            return 1
        else:
            # Run test matrix
            results = run_test_matrix(
                config=config,
                service_manager=service_manager,
                test_runner=test_runner,
                result_collector=result_collector,
                logger=logger
            )

            # Save raw results
            raw_results_file = output_dir / "all_results.json"
            with open(raw_results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved raw results to: {raw_results_file}")

            # Aggregate results
            logger.info("Aggregating results...")
            aggregated_results = result_collector.aggregate_results(results)

            # Generate reports
            logger.info("Generating reports...")

            if reporting_config.get('generate_html', True):
                html_report = report_generator.generate_html_report(aggregated_results, config)
                logger.info(f"HTML report: {html_report}")

            if reporting_config.get('generate_markdown', True):
                md_report = report_generator.generate_markdown_report(aggregated_results, config)
                logger.info(f"Markdown report: {md_report}")

            if reporting_config.get('generate_plots', True) and MATPLOTLIB_AVAILABLE:
                plot_files = report_generator.generate_plots(aggregated_results)
                logger.info(f"Generated {len(plot_files)} plots")

            # Copy configuration to output directory
            config_copy = output_dir / "test_config.yaml"
            shutil.copy(args.config, config_copy)
            logger.info(f"Copied configuration to: {config_copy}")

        logger.info("="*80)
        logger.info("Automated testing completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Cleanup: stop services
        try:
            logger.info("Stopping services...")
            service_manager = ServiceManager(script_dir, logger)
            service_manager.stop_services()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
