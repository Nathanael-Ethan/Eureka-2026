"""
Quick benchmark runner for CI / regression testing.

Runs a minimal set of benchmarks to detect regressions quickly.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ldmark.benchmark import (
    run_tensor_benchmark,
    run_repeatability_benchmark,
    run_prismml_q1_0_g128_benchmark,
    RegressionThresholds,
    run_regression_benchmarks,
    create_regression_suite,
    load_baseline,
)
from ldmark.benchmark.model import BenchmarkRecord, HardwareMetadata
import json


def run_quick_benchmarks() -> dict:
    """Run quick benchmark suite for regression detection."""
    results = {}
    
    # Test tensor
    tensor = np.random.default_rng(42).normal(0, 1, (1024, 1024)).astype(np.float32)
    
    # Quick method comparison
    methods = [
        ("FP32", 32, 1, np.float32),
        ("FP16", 16, 1, np.float16),
        ("INT8", 8, 128, np.float16),
        ("INT4", 4, 128, np.float16),
    ]
    
    print("Running quick benchmarks...")
    for method, target_bits, group_size, scale_dtype in methods:
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="quick_test",
            compression_method=method,
            target_bits=target_bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            random_seed=42,
        )
        
        results[method] = {
            "bits_per_weight": result.actual_bits_per_weight,
            "compression_ratio": result.compression_ratio,
            "mae": result.error_distribution.mean,
            "max_error": result.error_distribution.maximum,
            "relative_error": result.relative_error,
            "transform_time_ms": result.timing.transformation_time_ms,
            "dequant_time_ms": result.timing.dequantization_time_ms,
            "storage_overhead_ratio": result.storage.overhead_ratio,
        }
        print(f"  {method}: {result.actual_bits_per_weight:.4f} bpw, "
              f"{result.compression_ratio:.2f}x, MAE={result.error_distribution.mean:.6f}")
    
    # Repeatability check
    print("\nChecking repeatability...")
    repeat_results, analysis = run_repeatability_benchmark(
        tensor=tensor,
        tensor_name="repeatability",
        compression_method="INT8",
        target_bits=8,
        group_size=128,
        scale_dtype=np.float16,
        num_runs=5,
        base_seed=42,
    )
    results["repeatability"] = analysis
    print(f"  Identical: {analysis['all_results_identical']}")
    
    # PrismML validation
    print("\nValidating PrismML Q1_0_g128 mathematics...")
    prismml_result = run_prismml_q1_0_g128_benchmark(
        tensor=tensor,
        tensor_name="prismml_test",
    )
    results["prismml"] = {
        "theoretical_bpw": prismml_result.theoretical_bpw,
        "calculated_bpw": prismml_result.calculated_bpw,
        "bpw_error": prismml_result.bpw_error,
        "storage_validation_passed": prismml_result.storage_validation_passed,
        "mae": prismml_result.mae,
    }
    print(f"  Theoretical: {prismml_result.theoretical_bpw}, "
          f"Calculated: {prismml_result.calculated_bpw:.6f}, "
          f"Error: {prismml_result.bpw_error:.6f}")
    print(f"  Storage validation: {'PASSED' if prismml_result.storage_validation_passed else 'FAILED'}")
    
    return results


def run_regression_check(baseline_path: str = None) -> bool:
    """Run regression check against baseline."""
    if baseline_path is None:
        baseline_path = os.path.join(
            os.path.dirname(__file__), "results", "regression_baselines", "regression_baseline.json"
        )
    
    if not os.path.exists(baseline_path):
        print(f"Baseline not found at {baseline_path}, creating...")
        tensor_specs = [
            ("test_vector", (1024,), "float32"),
            ("test_matrix", (256, 256), "float32"),
        ]
        methods = [
            {"compression_method": "INT8", "target_bits": 8, "group_size": 128, "scale_dtype": "float16"},
            {"compression_method": "INT4", "target_bits": 4, "group_size": 128, "scale_dtype": "float16"},
        ]
        create_regression_suite(tensor_specs, methods, seeds=[42], output_dir=os.path.dirname(baseline_path))
        return True  # First run, no regression possible
    
    baseline = load_baseline(baseline_path)
    current_tensors = {
        "test_vector": np.random.default_rng(42).normal(0, 1, (1024,)).astype(np.float32),
        "test_matrix": np.random.default_rng(42).normal(0, 1, (256, 256)).astype(np.float32),
    }
    
    thresholds = RegressionThresholds(
        compression_ratio_threshold=0.05,
        storage_increase_threshold=0.05,
        error_regression_threshold=0.10,
        runtime_regression_threshold=0.20,
        bpw_increase_threshold=0.02,
        overhead_ratio_threshold=0.10,
    )
    
    result = run_regression_benchmarks(
        baseline_records=baseline,
        current_tensors=current_tensors,
        config={},
        thresholds=thresholds,
    )
    
    print(f"\nRegression Check: {'PASSED' if result.overall_passed else 'FAILED'}")
    print(f"  Checks: {result.summary['passed']}/{result.summary['total_checks']} passed")
    if result.summary['regressions_detected']:
        for reg in result.summary['regression_details']:
            print(f"  REGRESSION: {reg['check']} - baseline={reg['baseline']:.6f}, "
                  f"current={reg['current']:.6f}, ratio={reg['ratio']:.4f}")
    
    return result.overall_passed


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="LDMARK Quick Benchmark Runner")
    parser.add_argument("--regression", action="store_true", help="Run regression check")
    parser.add_argument("--baseline", type=str, help="Path to baseline JSON")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    if args.regression:
        passed = run_regression_check(args.baseline)
        sys.exit(0 if passed else 1)
    else:
        results = run_quick_benchmarks()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()