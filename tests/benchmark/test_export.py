"""
Tests for LDMARK Benchmark Laboratory - Export
"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from src.ldmark.benchmark.export import (
    export_benchmarks_csv,
    export_benchmarks_json,
    export_tensor_results_csv,
    export_model_results_csv,
    export_error_distributions_csv,
    export_comparison_table_csv,
    load_benchmarks_json,
    load_benchmarks_csv,
)
from src.ldmark.benchmark.model import (
    BenchmarkRecord,
    TensorBenchmarkResult,
    ModelBenchmarkResult,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
    HardwareMetadata,
)


def create_test_record() -> BenchmarkRecord:
    """Create a test benchmark record."""
    return BenchmarkRecord(
        experiment_id="test_exp_001",
        model_identifier="test_model",
        parameter_count=10000,
        tensor_count=2,
        source_dtype="float32",
        compression_method="INT8",
        target_bits=8,
        group_size=128,
        scale_dtype="float16",
        original_bytes=40000,
        compressed_bytes=10000,
        actual_bits_per_weight=8.125,
        compression_ratio=4.0,
        mae=0.01,
        mse=0.001,
        rmse=0.03,
        max_absolute_error=0.1,
        relative_error=0.05,
        error_distribution=ErrorDistribution(
            minimum=0.0, median=0.005, mean=0.01, maximum=0.1,
            std=0.02, p25=0.002, p75=0.015, p95=0.05, p99=0.08, count=10000
        ),
        transformation_time_ms=10.0,
        dequantization_time_ms=5.0,
        validation_time_ms=2.0,
        total_time_ms=17.0,
        theoretical_storage_bytes=10000,
        actual_storage_bytes=10200,
        metadata_bytes=0,
        scales_bytes=200,
        padding_bytes=0,
        tensor_headers_bytes=0,
        serialization_overhead_bytes=200,
        hardware_metadata=HardwareMetadata(
            platform="test", processor="test", python_version="test",
            numpy_version="test", cpu_count=4, total_memory_gb=16.0
        ),
        random_seed=42,
        timestamp="2024-01-01T00:00:00",
    )


def create_test_tensor_result() -> TensorBenchmarkResult:
    """Create a test tensor benchmark result."""
    return TensorBenchmarkResult(
        tensor_name="test_tensor",
        original_shape=(100, 100),
        original_dtype="float32",
        num_elements=10000,
        compression_method="INT8",
        target_bits=8,
        group_size=128,
        scale_dtype="float16",
        original_bytes=40000,
        compressed_bytes=10000,
        actual_bits_per_weight=8.125,
        compression_ratio=4.0,
        error_distribution=ErrorDistribution(
            minimum=0.0, median=0.005, mean=0.01, maximum=0.1,
            std=0.02, p25=0.002, p75=0.015, p95=0.05, p99=0.08, count=10000
        ),
        relative_error=0.05,
        timing=TimingMetrics(10.0, 5.0, 2.0, 17.0),
        storage=StorageMetrics(
            theoretical_bytes=10000, actual_bytes=10200,
            metadata_bytes=0, scales_bytes=200, padding_bytes=0,
            tensor_headers_bytes=0, serialization_overhead_bytes=200,
            overhead_ratio=1.02
        ),
        random_seed=42,
    )


def create_test_model_result() -> ModelBenchmarkResult:
    """Create a test model benchmark result."""
    tensor_result = create_test_tensor_result()
    return ModelBenchmarkResult(
        model_identifier="test_model",
        parameter_count=20000,
        tensor_count=2,
        tensor_results=[tensor_result, tensor_result],
        aggregate_compression_ratio=4.0,
        aggregate_bits_per_weight=8.125,
        aggregate_original_bytes=80000,
        aggregate_compressed_bytes=20000,
        aggregate_storage=StorageMetrics(
            theoretical_bytes=20000, actual_bytes=20400,
            metadata_bytes=0, scales_bytes=400, padding_bytes=0,
            tensor_headers_bytes=0, serialization_overhead_bytes=400,
            overhead_ratio=1.02
        ),
        total_timing=TimingMetrics(20.0, 10.0, 4.0, 34.0),
        random_seed=42,
    )


class TestExportBenchmarksCSV:
    def test_export_csv(self):
        records = [create_test_record()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_benchmarks_csv(records, filepath)
            
            # Verify file exists and has content
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "test_exp_001" in content
            assert "INT8" in content
            assert "8.125" in content
            assert "4.0" in content
        finally:
            os.unlink(filepath)
    
    def test_export_csv_multiple_records(self):
        records = [create_test_record(), create_test_record()]
        records[1] = BenchmarkRecord(
            **{**create_test_record().__dict__, "experiment_id": "test_exp_002", "compression_method": "INT4"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_benchmarks_csv(records, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "test_exp_001" in content
            assert "test_exp_002" in content
            assert "INT8" in content
            assert "INT4" in content
        finally:
            os.unlink(filepath)


class TestExportBenchmarksJSON:
    def test_export_json(self):
        records = [create_test_record()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            export_benchmarks_json(records, filepath)
            
            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["experiment_id"] == "test_exp_001"
            assert data[0]["compression_method"] == "INT8"
            assert "error_distribution" in data[0]
            assert "hardware_metadata" in data[0]
        finally:
            os.unlink(filepath)


class TestExportTensorResultsCSV:
    def test_export_tensor_csv(self):
        results = [create_test_tensor_result()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_tensor_results_csv(results, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "test_tensor" in content
            assert "INT8" in content
            assert "error_min" in content
            assert "error_p95" in content
        finally:
            os.unlink(filepath)


class TestExportModelResultsCSV:
    def test_export_model_csv(self):
        results = [create_test_model_result()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_model_results_csv(results, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "test_model" in content
            assert "20000" in content  # parameter_count
            assert "aggregate_compression_ratio" in content
        finally:
            os.unlink(filepath)


class TestExportErrorDistributionsCSV:
    def test_export_error_distributions(self):
        records = [create_test_record()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_error_distributions_csv(records, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "min" in content
            assert "median" in content
            assert "mean" in content
            assert "max" in content
            assert "p95" in content
            assert "p99" in content
            
            # Should have 9 rows (9 statistics) per record
            lines = content.strip().split('\n')
            assert len(lines) == 10  # Header + 9 statistics
        finally:
            os.unlink(filepath)


class TestExportComparisonTableCSV:
    def test_export_comparison_table(self):
        records = []
        for method in ["FP32", "FP16", "INT8", "INT4"]:
            record = create_test_record()
            record = BenchmarkRecord(
                **{**record.__dict__, "compression_method": method, "experiment_id": f"test_{method}"}
            )
            records.append(record)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_comparison_table_csv(records, filepath)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            assert "FP32" in content
            assert "FP16" in content
            assert "INT8" in content
            assert "INT4" in content
            assert "FP32_bits_per_weight" in content
            assert "INT8_compression_ratio" in content
        finally:
            os.unlink(filepath)


class TestLoadBenchmarks:
    def test_load_json(self):
        records = [create_test_record()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            export_benchmarks_json(records, filepath)
            loaded = load_benchmarks_json(filepath)
            
            assert len(loaded) == 1
            assert loaded[0].experiment_id == "test_exp_001"
            assert loaded[0].compression_method == "INT8"
        finally:
            os.unlink(filepath)
    
    def test_load_csv(self):
        records = [create_test_record()]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            filepath = f.name
        
        try:
            export_benchmarks_csv(records, filepath)
            loaded = load_benchmarks_csv(filepath)
            
            assert len(loaded) == 1
            assert loaded[0]["experiment_id"] == "test_exp_001"
            assert loaded[0]["compression_method"] == "INT8"
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])