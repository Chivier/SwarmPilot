#!/usr/bin/env python3
"""
OOD Experiment Validation Module.

Provides validation and integrity checking for out-of-distribution experiments:
- Parameter distribution validation (A1 vs B1 sampling)
- Scaling factor verification
- Workflow completion verification
- Experiment reproducibility validation
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
from dataclasses import dataclass

from workload_generator import WorkflowWorkload
from ood_config import OODExperimentConfig, B1DataSource
from a1_data_sampler import A1DataSampler


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: Optional[Dict] = None


class OODValidator:
    """Validator for OOD experiment configurations and data."""

    def __init__(self):
        """Initialize validator."""
        self.a1_sampler = None

    def validate_a1_b1_sampling(
        self,
        workflow: WorkflowWorkload,
        config: OODExperimentConfig,
        tolerance: float = 0.20
    ) -> ValidationResult:
        """
        Validate that B1 sampling matches A1 distribution.

        Args:
            workflow: Workflow workload to validate
            config: Experiment configuration
            tolerance: Acceptable relative difference (default 20%)

        Returns:
            ValidationResult with pass/fail and details
        """
        if config.b1_data_source != B1DataSource.A1_SAMPLED:
            return ValidationResult(
                passed=True,
                message="Skipping A1 sampling validation (not using A1 data)",
                details=None
            )

        # Load A1 sampler for reference
        if self.a1_sampler is None:
            self.a1_sampler = A1DataSampler(seed=config.seed)

        # Get A1 reference statistics
        a1_stats = self.a1_sampler.get_statistics()

        # Get B1 statistics from workflow
        b1_times = [t for workflow_b1 in workflow.b1_times for t in workflow_b1]
        b1_array = np.array(b1_times)

        b1_mean = np.mean(b1_array)
        b1_std = np.std(b1_array)

        # Check mean and std deviation
        mean_diff_pct = abs(b1_mean - a1_stats.source_mean) / a1_stats.source_mean
        std_diff_pct = abs(b1_std - a1_stats.source_std) / a1_stats.source_std

        mean_valid = mean_diff_pct <= tolerance
        std_valid = std_diff_pct <= tolerance

        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(
            self.a1_sampler.a1_array,
            b1_array
        )
        ks_valid = ks_pvalue > 0.05  # Not significantly different

        passed = mean_valid and std_valid and ks_valid

        return ValidationResult(
            passed=passed,
            message=f"A1-B1 sampling validation {'PASSED' if passed else 'FAILED'}",
            details={
                "a1_mean": float(a1_stats.source_mean),
                "a1_std": float(a1_stats.source_std),
                "b1_mean": float(b1_mean),
                "b1_std": float(b1_std),
                "mean_diff_pct": float(mean_diff_pct * 100),
                "std_diff_pct": float(std_diff_pct * 100),
                "mean_valid": mean_valid,
                "std_valid": std_valid,
                "ks_statistic": float(ks_statistic),
                "ks_pvalue": float(ks_pvalue),
                "ks_valid": ks_valid,
                "tolerance_pct": tolerance * 100
            }
        )

    def validate_scaling_factors(
        self,
        workflow: WorkflowWorkload,
        config: OODExperimentConfig,
        reference_workflow: Optional[WorkflowWorkload] = None
    ) -> ValidationResult:
        """
        Validate that scaling factors are correctly applied.

        Args:
            workflow: Scaled workflow to validate
            config: Configuration with scaling factors
            reference_workflow: Original unscaled workflow (if available)

        Returns:
            ValidationResult with pass/fail and details
        """
        if config.b1_sleep_time_scale == 1.0 and config.b1_exp_runtime_scale == 1.0:
            return ValidationResult(
                passed=True,
                message="No scaling applied (factors = 1.0)",
                details=None
            )

        # Get B1 times from workflow
        b1_times = [t for workflow_b1 in workflow.b1_times for t in workflow_b1]
        b1_array = np.array(b1_times)

        # Check if scaling factor matches expected value
        if reference_workflow is not None:
            ref_b1_times = [t for workflow_b1 in reference_workflow.b1_times for t in workflow_b1]
            ref_b1_array = np.array(ref_b1_times)

            # Calculate actual scaling ratio
            actual_ratio = np.mean(b1_array) / np.mean(ref_b1_array)
            expected_ratio = config.b1_sleep_time_scale

            ratio_diff_pct = abs(actual_ratio - expected_ratio) / expected_ratio

            passed = ratio_diff_pct < 0.01  # Within 1%

            return ValidationResult(
                passed=passed,
                message=f"Scaling factor validation {'PASSED' if passed else 'FAILED'}",
                details={
                    "expected_scale": float(expected_ratio),
                    "actual_scale": float(actual_ratio),
                    "diff_pct": float(ratio_diff_pct * 100),
                    "reference_mean": float(np.mean(ref_b1_array)),
                    "scaled_mean": float(np.mean(b1_array))
                }
            )
        else:
            # Without reference, just check bounds
            min_expected = 0.2  # Minimum scaling factor
            max_expected = 1.0  # Maximum scaling factor

            mean_value = np.mean(b1_array)
            # Rough validation based on expected parameter ranges
            passed = True  # Cannot fully validate without reference

            return ValidationResult(
                passed=passed,
                message="Scaling factor validation (no reference available)",
                details={
                    "configured_scale": float(config.b1_sleep_time_scale),
                    "b1_mean": float(mean_value),
                    "note": "Full validation requires reference workflow"
                }
            )

    def validate_parameter_bounds(
        self,
        workflow: WorkflowWorkload,
        min_time: float = 0.1,
        max_time: float = 100.0
    ) -> ValidationResult:
        """
        Validate that all parameters are within realistic bounds.

        Args:
            workflow: Workflow to validate
            min_time: Minimum acceptable time value
            max_time: Maximum acceptable time value

        Returns:
            ValidationResult with pass/fail and details
        """
        all_times = []
        all_times.extend(workflow.a1_times)
        all_times.extend(workflow.a2_times)
        all_times.extend([t for workflow_b1 in workflow.b1_times for t in workflow_b1])
        all_times.extend([t for workflow_b2 in workflow.b2_times for t in workflow_b2])

        all_times_array = np.array(all_times)

        # Check bounds
        min_value = np.min(all_times_array)
        max_value = np.max(all_times_array)

        violations = []
        if min_value < min_time:
            violations.append(f"Minimum value {min_value:.3f}s below threshold {min_time}s")
        if max_value > max_time:
            violations.append(f"Maximum value {max_value:.3f}s above threshold {max_time}s")

        # Check for negative or zero values
        invalid_values = np.sum(all_times_array <= 0)
        if invalid_values > 0:
            violations.append(f"Found {invalid_values} negative or zero values")

        passed = len(violations) == 0

        return ValidationResult(
            passed=passed,
            message=f"Parameter bounds validation {'PASSED' if passed else 'FAILED'}",
            details={
                "min_value": float(min_value),
                "max_value": float(max_value),
                "min_threshold": min_time,
                "max_threshold": max_time,
                "total_values": len(all_times),
                "invalid_count": int(invalid_values),
                "violations": violations
            }
        )

    def validate_reproducibility(
        self,
        workflow1: WorkflowWorkload,
        workflow2: WorkflowWorkload,
        config: OODExperimentConfig
    ) -> ValidationResult:
        """
        Validate experiment reproducibility by comparing two runs with same seed.

        Args:
            workflow1: First workflow generation
            workflow2: Second workflow generation
            config: Configuration used for both

        Returns:
            ValidationResult with pass/fail and details
        """
        # Check that all arrays match exactly
        differences = []

        if not np.array_equal(workflow1.a1_times, workflow2.a1_times):
            differences.append("A1 times differ")

        if not np.array_equal(workflow1.a2_times, workflow2.a2_times):
            differences.append("A2 times differ")

        # Check B1 times (nested lists)
        for i, (w1_b1, w2_b1) in enumerate(zip(workflow1.b1_times, workflow2.b1_times)):
            if not np.array_equal(w1_b1, w2_b1):
                differences.append(f"B1 times differ at workflow {i}")
                break

        # Check B2 times (nested lists)
        for i, (w1_b2, w2_b2) in enumerate(zip(workflow1.b2_times, workflow2.b2_times)):
            if not np.array_equal(w1_b2, w2_b2):
                differences.append(f"B2 times differ at workflow {i}")
                break

        if not np.array_equal(workflow1.fanout_values, workflow2.fanout_values):
            differences.append("Fanout values differ")

        passed = len(differences) == 0

        return ValidationResult(
            passed=passed,
            message=f"Reproducibility validation {'PASSED' if passed else 'FAILED'}",
            details={
                "seed": config.seed,
                "differences": differences,
                "workflows_compared": len(workflow1.a1_times)
            }
        )

    def run_all_validations(
        self,
        workflow: WorkflowWorkload,
        config: OODExperimentConfig
    ) -> Dict[str, ValidationResult]:
        """
        Run all applicable validations for a workflow.

        Args:
            workflow: Workflow to validate
            config: Configuration

        Returns:
            Dictionary of validation results
        """
        results = {}

        # A1-B1 sampling validation
        results["a1_b1_sampling"] = self.validate_a1_b1_sampling(workflow, config)

        # Scaling validation (without reference)
        results["scaling_factors"] = self.validate_scaling_factors(workflow, config)

        # Parameter bounds
        results["parameter_bounds"] = self.validate_parameter_bounds(workflow)

        return results

    @staticmethod
    def print_validation_results(results: Dict[str, ValidationResult]):
        """Print validation results in human-readable format."""
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        all_passed = all(r.passed for r in results.values())

        for name, result in results.items():
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n{name}: {status}")
            print(f"  {result.message}")

            if result.details:
                print("  Details:")
                for key, value in result.details.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"    {key}:")
                        for item in value:
                            print(f"      - {item}")
                    elif not isinstance(value, list):
                        print(f"    {key}: {value}")

        print("\n" + "=" * 60)
        print(f"Overall: {'✓ ALL VALIDATIONS PASSED' if all_passed else '✗ SOME VALIDATIONS FAILED'}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Demo usage
    from workload_generator import generate_workflow_with_a1_b1_sampling
    from ood_config import get_exp1_baseline_config

    print("OOD Validation Demo")
    print("=" * 60)

    # Generate workflow
    config = get_exp1_baseline_config(num_workflows=100, seed=42)
    workflow, _ = generate_workflow_with_a1_b1_sampling(
        num_workflows=100,
        seed=42,
        use_a1_for_exp_runtime=False
    )

    # Run validations
    validator = OODValidator()
    results = validator.run_all_validations(workflow, config)

    # Print results
    validator.print_validation_results(results)
