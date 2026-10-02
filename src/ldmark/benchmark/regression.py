"""
Regression benchmark suite for LDMARK.

Detects compression ratio regressions, storage increases, reconstruction error
regressions, and runtime regressions after LDMARK changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import numpy as np
from pathlib import Path
import json

from .model import BenchmarkRecord, TensorBenchmarkResult, ModelBenchmarkResult
from .runner import (
    run_tensor_benchmark,
    run_model_benchmark,
    run_repeatability_benchmark,
    create_synthetic_model,
)
from .storage import validate_storage, StorageValidationResult


@dataclass(frozen=True)
class RegressionThresholds:
    """Configurable thresholds for regression detection."""
    # Compression ratio regression: current must be >= baseline * (1 - threshold)
    compression_ratio_threshold: float = 0.05  # 5% degradation allowed
    
    # Storage increase: actual must be <= baseline * (1 + threshold)
    storage_increase_threshold: float = 0.05  # 5% increase allowed
    
    # Reconstruction error: current must be <= baseline * (1 + threshold)
    error_regression_threshold: float = 0.10  # 10% error increase allowed
    
    # Runtime: current must be <= baseline * (1 + threshold)
    runtime_regression_threshold: float = 0.20  # 20% slowdown allowed
    
    # Bits per weight: current must be <= baseline * (1 + threshold)
    bpw_increase_threshold: float = 0.02  # 2% increase allowed
    
    # Overhead ratio: current must be <= baseline * (1 + threshold)
    overhead_ratio_threshold: float = 0.10  # 10% overhead increase allowed


@dataclass(frozen=True)
class RegressionCheck:
    """Result of a single regression check."""
    check_name: str
    baseline_value: float
    current_value: float
    threshold: float
    passed: bool
    regression_detected: bool
    ratio: float  # current / baseline
    details: str = ""


@dataclass(frozen=True)
class RegressionResult:
    """Complete regression benchmark result."""
    experiment_id: str
    baseline_records: List[BenchmarkRecord]
    current_records: List[BenchmarkRecord]
    checks: List[RegressionCheck]
    overall_passed: bool
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "baseline_count": len(self.baseline_records),
            "current_count": len(self.current_records),
            "checks": [
                {
                    "check_name": c.check_name,
                    "baseline_value": c.baseline_value,
                    "current_value": c.current_value,
                    "threshold": c.threshold,
                    "passed": c.passed,
                    "regression_detected": c.regression_detected,
                    "ratio": c.ratio,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "overall_passed": self.overall_passed,
            "summary": self.summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def compare_records(
    baseline: BenchmarkRecord,
    current: BenchmarkRecord,
    thresholds: RegressionThresholds,
) -> List[RegressionCheck]:
    """
    Compare two benchmark records and detect regressions.
    
    Args:
        baseline: Baseline (expected) benchmark record
        current: Current benchmark record to check
        thresholds: Regression detection thresholds
        
    Returns:
        List of regression checks
    """
    checks = []
    
    # Compression ratio check
    if baseline.compression_ratio > 0:
        ratio = current.compression_ratio / baseline.compression_ratio
        passed = ratio >= (1 - thresholds.compression_ratio_threshold)
        checks.append(RegressionCheck(
            check_name="compression_ratio",
            baseline_value=baseline.compression_ratio,
            current_value=current.compression_ratio,
            threshold=thresholds.compression_ratio_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 - thresholds.compression_ratio_threshold:.4f}",
        ))
    
    # Bits per weight check
    if baseline.actual_bits_per_weight > 0:
        ratio = current.actual_bits_per_weight / baseline.actual_bits_per_weight
        passed = ratio <= (1 + thresholds.bpw_increase_threshold)
        checks.append(RegressionCheck(
            check_name="bits_per_weight",
            baseline_value=baseline.actual_bits_per_weight,
            current_value=current.actual_bits_per_weight,
            threshold=thresholds.bpw_increase_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.bpw_increase_threshold:.4f}",
        ))
    
    # Storage check
    if baseline.theoretical_storage_bytes > 0:
        ratio = current.actual_storage_bytes / baseline.actual_storage_bytes
        passed = ratio <= (1 + thresholds.storage_increase_threshold)
        checks.append(RegressionCheck(
            check_name="actual_storage",
            baseline_value=baseline.actual_storage_bytes,
            current_value=current.actual_storage_bytes,
            threshold=thresholds.storage_increase_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.storage_increase_threshold:.4f}",
        ))
    
    # MAE check
    if baseline.mae > 0:
        ratio = current.mae / baseline.mae
        passed = ratio <= (1 + thresholds.error_regression_threshold)
        checks.append(RegressionCheck(
            check_name="mae",
            baseline_value=baseline.mae,
            current_value=current.mae,
            threshold=thresholds.error_regression_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.error_regression_threshold:.4f}",
        ))
    
    # MSE check
    if baseline.mse > 0:
        ratio = current.mse / baseline.mse
        passed = ratio <= (1 + thresholds.error_regression_threshold)
        checks.append(RegressionCheck(
            check_name="mse",
            baseline_value=baseline.mse,
            current_value=current.mse,
            threshold=thresholds.error_regression_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.error_regression_threshold:.4f}",
        ))
    
    # Max absolute error check
    if baseline.max_absolute_error > 0:
        ratio = current.max_absolute_error / baseline.max_absolute_error
        passed = ratio <= (1 + thresholds.error_regression_threshold)
        checks.append(RegressionCheck(
            check_name="max_absolute_error",
            baseline_value=baseline.max_absolute_error,
            current_value=current.max_absolute_error,
            threshold=thresholds.error_regression_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.error_regression_threshold:.4f}",
        ))
    
    # Relative error check
    if baseline.relative_error > 0:
        ratio = current.relative_error / baseline.relative_error
        passed = ratio <= (1 + thresholds.error_regression_threshold)
        checks.append(RegressionCheck(
            check_name="relative_error",
            baseline_value=baseline.relative_error,
            current_value=current.relative_error,
            threshold=thresholds.error_regression_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.error_regression_threshold:.4f}",
        ))
    
    # Transformation time check
    if baseline.transformation_time_ms > 0:
        ratio = current.transformation_time_ms / baseline.transformation_time_ms
        passed = ratio <= (1 + thresholds.runtime_regression_threshold)
        checks.append(RegressionCheck(
            check_name="transformation_time",
            baseline_value=baseline.transformation_time_ms,
            current_value=current.transformation_time_ms,
            threshold=thresholds.runtime_regression_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.runtime_regression_threshold:.4f}",
        ))
    
    # Dequantization time check
    if baseline.dequantization_time_ms > 0:
        ratio = current.dequantization_time_ms / baseline.dequantization_time_ms
        passed = ratio <= (1 + thresholds.runtime_regression_threshold)
        checks.append(RegressionCheck(
            check_name="dequantization_time",
            baseline_value=baseline.dequantization_time_ms,
            current_value=current.dequantization_time_ms,
            threshold=thresholds.runtime_regression_threshold,
            passed=passed,
            regression_detected=not passed,
            ratio=ratio,
            details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.runtime_regression_threshold:.4f}",
        ))
    
    # Overhead ratio check
    if baseline.theoretical_storage_bytes > 0:
        baseline_overhead = baseline.actual_storage_bytes / baseline.theoretical_storage_bytes
        current_overhead = current.actual_storage_bytes / current.theoretical_storage_bytes
        if baseline_overhead > 0:
            ratio = current_overhead / baseline_overhead
            passed = ratio <= (1 + thresholds.overhead_ratio_threshold)
            checks.append(RegressionCheck(
                check_name="overhead_ratio",
                baseline_value=baseline_overhead,
                current_value=current_overhead,
                threshold=thresholds.overhead_ratio_threshold,
                passed=passed,
                regression_detected=not passed,
                ratio=ratio,
                details=f"Ratio {ratio:.4f}, threshold {1 + thresholds.overhead_ratio_threshold:.4f}",
            ))
    
    return checks


def run_regression_benchmarks(
    baseline_records: List[BenchmarkRecord],
    current_tensors: Dict[str, np.ndarray],
    config: Dict[str, Any],
    thresholds: Optional[RegressionThresholds] = None,
) -> RegressionResult:
    """
    Run regression benchmarks comparing current results against baseline.
    
    Args:
        baseline_records: List of baseline BenchmarkRecord objects
        current_tensors: Dictionary of tensors to benchmark
        config: Benchmark configuration matching baseline
        thresholds: Optional custom thresholds
        
    Returns:
        RegressionResult with all checks
    """
    if thresholds is None:
        thresholds = RegressionThresholds()
    
    # Run current benchmarks
    current_records = []
    for baseline in baseline_records:
        tensor = current_tensors.get(baseline.model_identifier)
        if tensor is None:
            # Try to find by tensor name in baseline
            # For now, assume single tensor or use first tensor
            tensor = list(current_tensors.values())[0]
        
        current = run_tensor_benchmark(
            tensor=tensor,
            tensor_name=baseline.model_identifier,
            compression_method=baseline.compression_method,
            target_bits=baseline.target_bits,
            group_size=baseline.group_size,
            scale_dtype=np.dtype(baseline.scale_dtype),
            random_seed=baseline.random_seed,
        )
        
        # Convert to BenchmarkRecord
        current_record = BenchmarkRecord(
            experiment_id=current.tensor_name,
            model_identifier=current.tensor_name,
            parameter_count=current.num_elements,
            tensor_count=1,
            source_dtype=current.original_dtype,
            compression_method=current.compression_method,
            target_bits=current.target_bits,
            group_size=current.group_size,
            scale_dtype=current.scale_dtype,
            original_bytes=current.original_bytes,
            compressed_bytes=current.compressed_bytes,
            actual_bits_per_weight=current.actual_bits_per_weight,
            compression_ratio=current.compression_ratio,
            mae=current.error_distribution.mean,  # Approximation
            mse=current.error_distribution.mean ** 2,  # Approximation
            rmse=current.error_distribution.mean,  # Approximation
            max_absolute_error=current.error_distribution.maximum,
            relative_error=current.relative_error,
            error_distribution=current.error_distribution,
            transformation_time_ms=current.timing.transformation_time_ms,
            dequantization_time_ms=current.timing.dequantization_time_ms,
            validation_time_ms=current.timing.validation_time_ms,
            total_time_ms=current.timing.total_time_ms,
            theoretical_storage_bytes=current.storage.theoretical_bytes,
            actual_storage_bytes=current.storage.actual_bytes,
            metadata_bytes=current.storage.metadata_bytes,
            scales_bytes=current.storage.scales_bytes,
            padding_bytes=current.storage.padding_bytes,
            tensor_headers_bytes=current.storage.tensor_headers_bytes,
            serialization_overhead_bytes=current.storage.serialization_overhead_bytes,
            hardware_metadata=baseline.hardware_metadata,
            random_seed=current.random_seed,
            timestamp=baseline.timestamp,
        )
        current_records.append(current_record)
    
    # Compare each pair
    all_checks = []
    for baseline, current in zip(baseline_records, current_records):
        checks = compare_records(baseline, current, thresholds)
        all_checks.extend(checks)
    
    # Overall result
    overall_passed = all(c.passed for c in all_checks)
    regressions = [c for c in all_checks if c.regression_detected]
    
    summary = {
        "total_checks": len(all_checks),
        "passed": sum(1 for c in all_checks if c.passed),
        "failed": len(regressions),
        "regressions_detected": len(regressions) > 0,
        "regression_details": [
            {
                "check": r.check_name,
                "baseline": r.baseline_value,
                "current": r.current_value,
                "ratio": r.ratio,
            }
            for r in regressions
        ],
    }
    
    return RegressionResult(
        experiment_id=f"regression_{len(baseline_records)}_checks",
        baseline_records=baseline_records,
        current_records=current_records,
        checks=all_checks,
        overall_passed=overall_passed,
        summary=summary,
    )


def create_regression_suite(
    tensor_specs: List[tuple],
    methods: List[Dict[str, Any]],
    seeds: List[int] = None,
    output_dir: str = "benchmark_baselines",
) -> List[BenchmarkRecord]:
    """
    Create a baseline regression suite.
    
    Runs benchmarks on synthetic tensors with specified configurations
    and saves results as baseline for future regression testing.
    
    Args:
        tensor_specs: List of (name, shape, dtype) tuples
        methods: List of method configs with keys:
            - compression_method
            - target_bits
            - group_size
            - scale_dtype
        seeds: Random seeds to use
        output_dir: Directory to save baseline JSON files
        
    Returns:
        List of baseline BenchmarkRecord objects
    """
    if seeds is None:
        seeds = [42]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    for seed in seeds:
        for name, shape, dtype_str in tensor_specs:
            tensors = create_synthetic_model(
                tensor_specs=[(name, shape, dtype_str)],
                generator="random_normal",
                seed=seed,
            )
            tensor = tensors[name]
            
            for method in methods:
                result = run_tensor_benchmark(
                    tensor=tensor,
                    tensor_name=name,
                    compression_method=method["compression_method"],
                    target_bits=method["target_bits"],
                    group_size=method["group_size"],
                    scale_dtype=np.dtype(method["scale_dtype"]),
                    random_seed=seed,
                )
                
                record = BenchmarkRecord(
                    experiment_id=f"{name}_{method['compression_method']}_g{method['group_size']}_seed{seed}",
                    model_identifier=name,
                    parameter_count=tensor.size,
                    tensor_count=1,
                    source_dtype=str(tensor.dtype),
                    compression_method=method["compression_method"],
                    target_bits=method["target_bits"],
                    group_size=method["group_size"],
                    scale_dtype=method["scale_dtype"],
                    original_bytes=result.original_bytes,
                    compressed_bytes=result.compressed_bytes,
                    actual_bits_per_weight=result.actual_bits_per_weight,
                    compression_ratio=result.compression_ratio,
                    mae=result.error_distribution.mean,
                    mse=result.error_distribution.mean ** 2,
                    rmse=result.error_distribution.mean,
                    max_absolute_error=result.error_distribution.maximum,
                    relative_error=result.relative_error,
                    error_distribution=result.error_distribution,
                    transformation_time_ms=result.timing.transformation_time_ms,
                    dequantization_time_ms=result.timing.dequantization_time_ms,
                    validation_time_ms=result.timing.validation_time_ms,
                    total_time_ms=result.timing.total_time_ms,
                    theoretical_storage_bytes=result.storage.theoretical_bytes,
                    actual_storage_bytes=result.storage.actual_bytes,
                    metadata_bytes=result.storage.metadata_bytes,
                    scales_bytes=result.storage.scales_bytes,
                    padding_bytes=result.storage.padding_bytes,
                    tensor_headers_bytes=result.storage.tensor_headers_bytes,
                    serialization_overhead_bytes=result.storage.serialization_overhead_bytes,
                    hardware_metadata=result.__dict__.get('hardware_metadata', None),
                    random_seed=seed,
                    timestamp="",
                )
                all_records.append(record)
    
    # Save baseline
    baseline_path = Path(output_dir) / "regression_baseline.json"
    with open(baseline_path, 'w') as f:
        json.dump([r.to_dict() for r in all_records], f, indent=2)
    
    return all_records


def load_baseline(filepath: str) -> List[BenchmarkRecord]:
    """Load baseline records from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [BenchmarkRecord.from_dict(d) for d in data]


def run_regression_suite(
    baseline_path: str,
    current_tensors: Dict[str, np.ndarray],
    thresholds: Optional[RegressionThresholds] = None,
) -> RegressionResult:
    """
    Run full regression suite against saved baseline.
    
    Args:
        baseline_path: Path to baseline JSON file
        current_tensors: Current tensors to test
        thresholds: Optional custom thresholds
        
    Returns:
        RegressionResult
    """
    baseline = load_baseline(baseline_path)
    
    # Group baseline by model_identifier and config
    # For simplicity, run against each baseline record
    config = {}  # Would extract from baseline[0] in real usage
    
    return run_regression_benchmarks(
        baseline_records=baseline,
        current_tensors=current_tensors,
        config=config,
        thresholds=thresholds,
    )