"""
Tests for LDMARK Benchmark Laboratory - Regression
"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from src.ldmark.benchmark.regression import (
    RegressionThresholds,
    RegressionCheck,
    RegressionResult,
    compare_records,
    run_regression_benchmarks,
    create_regression_suite,
    load_baseline,
)
from src.ldmark.benchmark.model import (
    BenchmarkRecord,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
    HardwareMetadata,
)
from src.ldmark.benchmark.runner import create_synthetic_model


def create_test_record(**kwargs) -> BenchmarkRecord:
    """Create a test benchmark record with defaults."""
    defaults = {
        "experiment_id": "test_001",
        "model_identifier": "test_model",
        "parameter_count": 10000,
        "tensor_count": 1,
        "source_dtype": "float32",
        "compression_method": "INT8",
        "target_bits": 8,
        "group_size": 128,
        "scale_dtype": "float16",
        "original_bytes": 40000,
        "compressed_bytes": 10000,
        "actual_bits_per_weight": 8.125,
        "compression_ratio": 4.0,
        "mae": 0.01,
        "mse": 0.001,
        "rmse": 0.03,
        "max_absolute_error": 0.1,
        "relative_error": 0.05,
        "error_distribution": ErrorDistribution(
            minimum=0.0, median=0.005, mean=0.01, maximum=0.1,
            std=0.02, p25=0.002, p75=0.015, p95=0.05, p99=0.08, count=10000
        ),
        "transformation_time_ms": 10.0,
        "dequantization_time_ms": 5.0,
        "validation_time_ms": 2.0,
        "total_time_ms": 17.0,
        "theoretical_storage_bytes": 10000,
        "actual_storage_bytes": 10200,
        "metadata_bytes": 0,
        "scales_bytes": 200,
        "padding_bytes": 0,
        "tensor_headers_bytes": 0,
        "serialization_overhead_bytes": 200,
        "hardware_metadata": HardwareMetadata(
            platform="test", processor="test", python_version="test",
            numpy_version="test", cpu_count=4, total_memory_gb=16.0
        ),
        "random_seed": 42,
        "timestamp": "2024-01-01T00:00:00",
    }
    defaults.update(kwargs)
    return BenchmarkRecord(**defaults)


class TestRegressionThresholds:
    def test_default_thresholds(self):
        thresholds = RegressionThresholds()
        assert thresholds.compression_ratio_threshold == 0.05
        assert thresholds.storage_increase_threshold == 0.05
        assert thresholds.error_regression_threshold == 0.10
        assert thresholds.runtime_regression_threshold == 0.20
        assert thresholds.bpw_increase_threshold == 0.02
        assert thresholds.overhead_ratio_threshold == 0.10
    
    def test_custom_thresholds(self):
        thresholds = RegressionThresholds(
            compression_ratio_threshold=0.01,
            error_regression_threshold=0.05,
        )
        assert thresholds.compression_ratio_threshold == 0.01
        assert thresholds.error_regression_threshold == 0.05


class TestCompareRecords:
    def test_no_regression(self):
        baseline = create_test_record()
        current = create_test_record()  # Identical
        
        thresholds = RegressionThresholds()
        checks = compare_records(baseline, current, thresholds)
        
        assert len(checks) > 0
        for check in checks:
            assert check.passed is True
            assert check.regression_detected is False
            assert check.ratio == 1.0
    
    def test_compression_ratio_regression(self):
        baseline = create_test_record(compression_ratio=4.0)
        current = create_test_record(compression_ratio=3.7)  # ~7.5% degradation
        
        thresholds = RegressionThresholds(compression_ratio_threshold=0.05)
        checks = compare_records(baseline, current, thresholds)
        
        cr_check = next(c for c in checks if c.check_name == "compression_ratio")
        assert cr_check.regression_detected is True
        assert cr_check.passed is False
    
    def test_compression_ratio_within_threshold(self):
        baseline = create_test_record(compression_ratio=4.0)
        current = create_test_record(compression_ratio=3.85)  # ~3.75% degradation
        
        thresholds = RegressionThresholds(compression_ratio_threshold=0.05)
        checks = compare_records(baseline, current, thresholds)
        
        cr_check = next(c for c in checks if c.check_name == "compression_ratio")
        assert cr_check.regression_detected is False
        assert cr_check.passed is True
    
    def test_bits_per_weight_regression(self):
        baseline = create_test_record(actual_bits_per_weight=8.125)
        current = create_test_record(actual_bits_per_weight=8.3)  # ~2.15% increase
        
        thresholds = RegressionThresholds(bpw_increase_threshold=0.02)
        checks = compare_records(baseline, current, thresholds)
        
        bpw_check = next(c for c in checks if c.check_name == "bits_per_weight")
        assert bpw_check.regression_detected is True
    
    def test_error_regression(self):
        baseline = create_test_record(mae=0.01, mse=0.001, max_absolute_error=0.1, relative_error=0.05)
        current = create_test_record(mae=0.012, mse=0.0015, max_absolute_error=0.12, relative_error=0.06)  # 20% increase
        
        thresholds = RegressionThresholds(error_regression_threshold=0.10)
        checks = compare_records(baseline, current, thresholds)
        
        for check in checks:
            if check.check_name in ("mae", "mse", "max_absolute_error", "relative_error"):
                assert check.regression_detected is True
    
    def test_runtime_regression(self):
        baseline = create_test_record(transformation_time_ms=10.0, dequantization_time_ms=5.0)
        current = create_test_record(transformation_time_ms=13.0, dequantization_time_ms=6.5)  # 30% increase
        
        thresholds = RegressionThresholds(runtime_regression_threshold=0.20)
        checks = compare_records(baseline, current, thresholds)
        
        for check in checks:
            if check.check_name in ("transformation_time", "dequantization_time"):
                assert check.regression_detected is True
    
    def test_storage_increase_regression(self):
        baseline = create_test_record(actual_storage_bytes=10200)
        current = create_test_record(actual_storage_bytes=11000)  # ~7.8% increase
        
        thresholds = RegressionThresholds(storage_increase_threshold=0.05)
        checks = compare_records(baseline, current, thresholds)
        
        storage_check = next(c for c in checks if c.check_name == "actual_storage")
        assert storage_check.regression_detected is True


class TestCreateRegressionSuite:
    def test_create_suite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_specs = [
                ("test_vec", (1024,), "float32"),
                ("test_mat", (64, 64), "float32"),
            ]
            methods = [
                {"compression_method": "INT8", "target_bits": 8, "group_size": 128, "scale_dtype": "float16"},
                {"compression_method": "INT4", "target_bits": 4, "group_size": 128, "scale_dtype": "float16"},
            ]
            
            baseline = create_regression_suite(
                tensor_specs=tensor_specs,
                methods=methods,
                seeds=[42],
                output_dir=tmpdir,
            )
            
            # Should have 2 tensors * 2 methods * 1 seed = 4 records
            assert len(baseline) == 4
            
            # Check baseline file created
            baseline_path = os.path.join(tmpdir, "regression_baseline.json")
            assert os.path.exists(baseline_path)
            
            # Verify content
            with open(baseline_path, 'r') as f:
                data = json.load(f)
            assert len(data) == 4
    
    def test_load_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_specs = [("test", (100,), "float32")]
            methods = [{"compression_method": "INT8", "target_bits": 8, "group_size": 128, "scale_dtype": "float16"}]
            
            create_regression_suite(tensor_specs, methods, seeds=[42], output_dir=tmpdir)
            
            baseline_path = os.path.join(tmpdir, "regression_baseline.json")
            loaded = load_baseline(baseline_path)
            
            assert len(loaded) == 1
            assert loaded[0].compression_method == "INT8"


class TestRegressionResult:
    def test_to_json(self):
        baseline = create_test_record()
        current = create_test_record()
        
        check = RegressionCheck(
            check_name="test_check",
            baseline_value=1.0,
            current_value=1.0,
            threshold=0.05,
            passed=True,
            regression_detected=False,
            ratio=1.0,
        )
        
        result = RegressionResult(
            experiment_id="test_regression",
            baseline_records=[baseline],
            current_records=[current],
            checks=[check],
            overall_passed=True,
            summary={"total_checks": 1, "passed": 1, "failed": 0},
        )
        
        json_str = result.to_json()
        assert "test_regression" in json_str
        assert "test_check" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])