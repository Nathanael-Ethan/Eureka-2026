"""
Tests for LDMARK Compute Abstraction
"""

from __future__ import annotations

import numpy as np
import pytest

from ldmark.compute import (
    ComputationConfig,
    ComputePrecision,
    MemoryStats,
    TimingStats,
    ComputeResult,
    create_reference_result,
    compare_results,
)
from ldmark.kernels import (
    Int8Matrix,
    int8_matmul_direct,
    int8_matmul_reference,
)
from ldmark.compression.quantize import (
    quantize_int8,
    QuantizationConfig,
    QuantizationTarget,
)


def create_test_matrix(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


class TestComputationConfig:
    def test_default_config(self):
        config = ComputationConfig()
        assert config.output_precision == ComputePrecision.FP32
        assert config.accumulate_in_fp32 is True
        assert config.batch_size == 1
        assert config.enable_memory_tracking is True
    
    def test_custom_config(self):
        config = ComputationConfig(
            output_precision=ComputePrecision.FP16,
            accumulate_in_fp32=False,
            batch_size=4,
            enable_memory_tracking=False,
        )
        assert config.output_precision == ComputePrecision.FP16
        assert config.accumulate_in_fp32 is False
        assert config.batch_size == 4
        assert config.enable_memory_tracking is False


class TestMemoryStats:
    def test_memory_stats_creation(self):
        stats = MemoryStats(
            compressed_weight_bytes=1000,
            decompressed_weight_bytes=4000,
            input_bytes=500,
            output_bytes=200,
            temporary_bytes=100,
            peak_bytes=2000,
        )
        assert stats.compressed_weight_bytes == 1000
        assert stats.decompressed_weight_bytes == 4000
        
        d = stats.to_dict()
        assert d["compressed_weight_bytes"] == 1000


class TestTimingStats:
    def test_timing_stats_creation(self):
        stats = TimingStats(
            preparation_time_s=0.1,
            computation_time_s=0.5,
            total_time_s=0.6,
        )
        assert stats.preparation_time_s == 0.1
        assert stats.computation_time_s == 0.5
        assert stats.total_time_s == 0.6
        
        d = stats.to_dict()
        assert d["preparation_time_s"] == 0.1


class TestComputeResult:
    def test_compute_result_creation(self):
        output = np.random.randn(10, 5).astype(np.float32)
        result = ComputeResult(
            output=output,
            memory_stats=MemoryStats(),
            timing_stats=TimingStats(),
        )
        assert result.output.shape == (10, 5)
        assert result.metadata == {}


class TestCompareResults:
    def test_compare_identical(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a.copy()
        
        result = compare_results(
            ComputeResult(output=a, memory_stats=MemoryStats(), timing_stats=TimingStats()),
            ComputeResult(output=b, memory_stats=MemoryStats(), timing_stats=TimingStats()),
        )
        assert result["passed"] is True
        assert result["max_absolute_error"] == 0.0
        assert result["max_relative_error"] == 0.0
    
    def test_compare_small_difference(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a + 1e-6
        
        result = compare_results(
            ComputeResult(output=a, memory_stats=MemoryStats(), timing_stats=TimingStats()),
            ComputeResult(output=b, memory_stats=MemoryStats(), timing_stats=TimingStats()),
            rtol=1e-5,
            atol=1e-7,
        )
        assert result["passed"] is True
        assert result["max_absolute_error"] <= 1e-6
    
    def test_compare_large_difference(self):
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[100.0, 2.0]], dtype=np.float32)
        
        result = compare_results(
            ComputeResult(output=a, memory_stats=MemoryStats(), timing_stats=TimingStats()),
            ComputeResult(output=b, memory_stats=MemoryStats(), timing_stats=TimingStats()),
        )
        assert result["passed"] is False
        assert result["max_absolute_error"] > 10


class TestCreateReferenceResult:
    def test_reference_result_creation(self):
        class MockWeight:
            def __init__(self):
                self.compressed_size_bytes = 100
                self.decompressed_size_bytes = 400
            def dequantize(self):
                return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        
        weight = MockWeight()
        x = np.array([[1.0], [2.0]], dtype=np.float32)
        config = ComputationConfig()
        
        result = create_reference_result(weight, x, config)
        
        assert isinstance(result, ComputeResult)
        assert result.output.shape == (2, 1)
        assert result.output[0, 0] == 5.0  # 1*1 + 2*2
        assert result.output[1, 0] == 11.0  # 3*1 + 4*2
        assert result.metadata["method"] == "reference_dequantize_then_matmul"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])