"""
Tests for LDMARK Benchmark Laboratory - Data Model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest
import json

from src.ldmark.benchmark.model import (
    HardwareMetadata,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
    TensorBenchmarkResult,
    ModelBenchmarkResult,
    BenchmarkRecord,
    BenchmarkConfig,
)


class TestHardwareMetadata:
    def test_capture(self):
        hw = HardwareMetadata.capture()
        assert hw.platform is not None
        assert hw.processor is not None
        assert hw.python_version is not None
        assert hw.numpy_version is not None
        assert hw.cpu_count >= 0
        assert hw.timestamp is not None
    
    def test_to_dict(self):
        hw = HardwareMetadata(
            platform="test",
            processor="test",
            python_version="test",
            numpy_version="test",
            cpu_count=4,
            total_memory_gb=16.0,
        )
        d = hw.to_dict()
        assert d["platform"] == "test"
        assert d["cpu_count"] == 4


class TestErrorDistribution:
    def test_from_errors_basic(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = ErrorDistribution.from_errors(errors)
        assert dist.minimum == 1.0
        assert dist.maximum == 5.0
        assert dist.mean == 3.0
        assert dist.median == 3.0
        assert dist.count == 5
    
    def test_from_errors_empty(self):
        errors = np.array([])
        dist = ErrorDistribution.from_errors(errors)
        assert dist.count == 0
        assert dist.minimum == 0.0
    
    def test_percentiles(self):
        errors = np.arange(1, 101, dtype=np.float64)
        dist = ErrorDistribution.from_errors(errors)
        assert dist.p25 == 25.75  # numpy percentile
        assert dist.p75 == 75.25
        assert dist.p95 == 95.05
        assert dist.p99 == 99.01


class TestTimingMetrics:
    def test_creation(self):
        timing = TimingMetrics(
            transformation_time_ms=10.0,
            dequantization_time_ms=5.0,
            validation_time_ms=2.0,
            total_time_ms=17.0,
        )
        assert timing.transformation_time_ms == 10.0
        assert timing.total_time_ms == 17.0


class TestStorageMetrics:
    def test_creation(self):
        storage = StorageMetrics(
            theoretical_bytes=1000,
            actual_bytes=1100,
            metadata_bytes=50,
            scales_bytes=50,
            padding_bytes=0,
            tensor_headers_bytes=0,
            serialization_overhead_bytes=100,
            overhead_ratio=1.1,
        )
        assert storage.theoretical_bytes == 1000
        assert storage.overhead_ratio == 1.1


class TestTensorBenchmarkResult:
    def test_creation(self):
        result = TensorBenchmarkResult(
            tensor_name="test_tensor",
            original_shape=(64, 64),
            original_dtype="float32",
            num_elements=4096,
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            original_bytes=16384,
            compressed_bytes=4096,
            actual_bits_per_weight=8.125,
            compression_ratio=4.0,
            error_distribution=ErrorDistribution(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            relative_error=0.01,
            timing=TimingMetrics(1.0, 1.0, 0.5, 2.5),
            storage=StorageMetrics(4000, 4096, 0, 96, 0, 0, 96, 1.024),
            random_seed=42,
        )
        assert result.tensor_name == "test_tensor"
        assert result.compression_method == "INT8"
        assert result.actual_bits_per_weight == 8.125
    
    def test_to_dict(self):
        result = TensorBenchmarkResult(
            tensor_name="test",
            original_shape=(10,),
            original_dtype="float32",
            num_elements=10,
            compression_method="INT8",
            target_bits=8,
            group_size=10,
            scale_dtype="float16",
            original_bytes=40,
            compressed_bytes=12,
            actual_bits_per_weight=9.6,
            compression_ratio=3.33,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,10),
            relative_error=0.1,
            timing=TimingMetrics(1,1,1,3),
            storage=StorageMetrics(10,12,0,2,0,0,2,1.2),
            random_seed=42,
        )
        d = result.to_dict()
        assert d["tensor_name"] == "test"
        assert "error_distribution" in d
        assert "timing" in d
        assert "storage" in d


class TestModelBenchmarkResult:
    def test_creation(self):
        tensor_result = TensorBenchmarkResult(
            tensor_name="t1",
            original_shape=(10,),
            original_dtype="float32",
            num_elements=10,
            compression_method="INT8",
            target_bits=8,
            group_size=10,
            scale_dtype="float16",
            original_bytes=40,
            compressed_bytes=12,
            actual_bits_per_weight=9.6,
            compression_ratio=3.33,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,10),
            relative_error=0.1,
            timing=TimingMetrics(1,1,1,3),
            storage=StorageMetrics(10,12,0,2,0,0,2,1.2),
            random_seed=42,
        )
        
        model_result = ModelBenchmarkResult(
            model_identifier="test_model",
            parameter_count=20,
            tensor_count=2,
            tensor_results=[tensor_result, tensor_result],
            aggregate_compression_ratio=3.33,
            aggregate_bits_per_weight=9.6,
            aggregate_original_bytes=80,
            aggregate_compressed_bytes=24,
            aggregate_storage=StorageMetrics(20,24,0,4,0,0,4,1.2),
            total_timing=TimingMetrics(2,2,2,6),
            random_seed=42,
        )
        
        assert model_result.model_identifier == "test_model"
        assert model_result.parameter_count == 20
        assert model_result.tensor_count == 2
        assert len(model_result.tensor_results) == 2


class TestBenchmarkRecord:
    def test_creation(self):
        record = BenchmarkRecord(
            experiment_id="test_123",
            model_identifier="test_model",
            parameter_count=1000,
            tensor_count=1,
            source_dtype="float32",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            original_bytes=4000,
            compressed_bytes=1000,
            actual_bits_per_weight=8.125,
            compression_ratio=4.0,
            mae=0.01,
            mse=0.001,
            rmse=0.03,
            max_absolute_error=0.1,
            relative_error=0.05,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,1000),
            transformation_time_ms=10.0,
            dequantization_time_ms=5.0,
            validation_time_ms=2.0,
            total_time_ms=17.0,
            theoretical_storage_bytes=1000,
            actual_storage_bytes=1050,
            metadata_bytes=0,
            scales_bytes=50,
            padding_bytes=0,
            tensor_headers_bytes=0,
            serialization_overhead_bytes=50,
            hardware_metadata=HardwareMetadata.capture(),
            random_seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        
        assert record.experiment_id == "test_123"
        assert record.compression_ratio == 4.0
        assert record.actual_bits_per_weight == 8.125
    
    def test_to_json(self):
        record = BenchmarkRecord(
            experiment_id="test",
            model_identifier="model",
            parameter_count=100,
            tensor_count=1,
            source_dtype="float32",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            original_bytes=400,
            compressed_bytes=100,
            actual_bits_per_weight=8.0,
            compression_ratio=4.0,
            mae=0.0,
            mse=0.0,
            rmse=0.0,
            max_absolute_error=0.0,
            relative_error=0.0,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,100),
            transformation_time_ms=1.0,
            dequantization_time_ms=1.0,
            validation_time_ms=1.0,
            total_time_ms=3.0,
            theoretical_storage_bytes=100,
            actual_storage_bytes=100,
            metadata_bytes=0,
            scales_bytes=0,
            padding_bytes=0,
            tensor_headers_bytes=0,
            serialization_overhead_bytes=0,
            hardware_metadata=HardwareMetadata.capture(),
            random_seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        
        json_str = record.to_json()
        assert "test" in json_str
        assert "INT8" in json_str
    
    def test_from_json(self):
        record = BenchmarkRecord(
            experiment_id="test",
            model_identifier="model",
            parameter_count=100,
            tensor_count=1,
            source_dtype="float32",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            original_bytes=400,
            compressed_bytes=100,
            actual_bits_per_weight=8.0,
            compression_ratio=4.0,
            mae=0.0,
            mse=0.0,
            rmse=0.0,
            max_absolute_error=0.0,
            relative_error=0.0,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,100),
            transformation_time_ms=1.0,
            dequantization_time_ms=1.0,
            validation_time_ms=1.0,
            total_time_ms=3.0,
            theoretical_storage_bytes=100,
            actual_storage_bytes=100,
            metadata_bytes=0,
            scales_bytes=0,
            padding_bytes=0,
            tensor_headers_bytes=0,
            serialization_overhead_bytes=0,
            hardware_metadata=HardwareMetadata.capture(),
            random_seed=42,
            timestamp="2024-01-01T00:00:00",
        )
        
        json_str = record.to_json()
        loaded = BenchmarkRecord.from_json(json_str)
        assert loaded.experiment_id == "test"
        assert loaded.compression_method == "INT8"


class TestBenchmarkConfig:
    def test_creation(self):
        config = BenchmarkConfig(
            experiment_name="test_exp",
            model_identifier="test_model",
            source_dtype=np.float32,
            compression_methods=["INT8", "INT4"],
            target_bits_list=[8, 4],
            group_sizes=[128, 64],
            scale_dtypes=[np.float16],
            tensor_shapes=[(64, 64)],
            tensor_generators=["random_normal"],
            random_seeds=[42],
        )
        
        assert config.experiment_name == "test_exp"
        assert len(config.compression_methods) == 2
        assert config.source_dtype == np.float32
    
    def test_to_dict(self):
        config = BenchmarkConfig(
            experiment_name="test",
            model_identifier="model",
            source_dtype=np.float32,
            compression_methods=["INT8"],
            target_bits_list=[8],
            group_sizes=[128],
            scale_dtypes=[np.float16],
            tensor_shapes=[(10,)],
            tensor_generators=["random_normal"],
            random_seeds=[42],
        )
        
        d = config.to_dict()
        assert d["source_dtype"] == "float32"
        assert "float16" in d["scale_dtypes"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])