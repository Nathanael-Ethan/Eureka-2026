"""
Tests for LDMARK Benchmark Laboratory - Storage Validation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from src.ldmark.benchmark.storage import (
    calculate_theoretical_storage,
    calculate_actual_storage,
    validate_storage,
    validate_model_storage,
    estimate_serialized_size,
    StorageValidationResult,
)


class TestCalculateTheoreticalStorage:
    def test_fp32_no_grouping(self):
        # 1000 weights at 32 bits, no grouping
        size = calculate_theoretical_storage(1000, 32, 0)
        assert size == 4000  # 1000 * 32 / 8
    
    def test_fp16_no_grouping(self):
        size = calculate_theoretical_storage(1000, 16, 0)
        assert size == 2000  # 1000 * 16 / 8
    
    def test_int8_grouped(self):
        # 1000 weights at 8 bits, groups of 128, 16-bit scales
        size = calculate_theoretical_storage(1000, 8, 128, 16)
        # Groups: ceil(1000/128) = 8
        # Bits: 1000*8 + 8*16 = 8000 + 128 = 8128
        # Bytes: ceil(8128/8) = 1016
        assert size == 1016
    
    def test_int4_grouped(self):
        # 1000 weights at 4 bits, groups of 128, 16-bit scales
        size = calculate_theoretical_storage(1000, 4, 128, 16)
        # Groups: 8
        # Bits: 1000*4 + 8*16 = 4000 + 128 = 4128
        # Bytes: ceil(4128/8) = 516
        assert size == 516
    
    def test_binary_q1_0_g128(self):
        # 128 weights at 1 bit, 1 group, 16-bit scale
        size = calculate_theoretical_storage(128, 1, 128, 16)
        # 128*1 + 1*16 = 144 bits = 18 bytes
        assert size == 18
    
    def test_large_model(self):
        # 27B parameters at 1.125 bits/weight (Q1_0_g128)
        size = calculate_theoretical_storage(27_000_000_000, 1, 128, 16)
        # 27e9 * 1.125 / 8 = ~3.79 GB
        expected_gb = 27_000_000_000 * 1.125 / 8 / (1024**3)
        actual_gb = size / (1024**3)
        assert abs(actual_gb - expected_gb) < 0.01


class TestCalculateActualStorage:
    def test_basic(self):
        quantized = np.zeros(1000, dtype=np.int8)
        scales = np.zeros(8, dtype=np.float16)
        
        result = calculate_actual_storage(quantized, scales)
        
        assert result["data_bytes"] == 1000
        assert result["scales_bytes"] == 16
        assert result["total_bytes"] == 1016
    
    def test_no_scales(self):
        quantized = np.zeros(1000, dtype=np.uint8)  # Packed INT4
        result = calculate_actual_storage(quantized, None)
        
        assert result["data_bytes"] == 1000
        assert result["scales_bytes"] == 0
        assert result["total_bytes"] == 1000
    
    def test_with_metadata(self):
        quantized = np.zeros(100, dtype=np.int8)
        scales = np.zeros(1, dtype=np.float16)
        
        result = calculate_actual_storage(
            quantized, scales,
            metadata_bytes=64,
            tensor_header_bytes=32,
            padding_bytes=4,
        )
        
        assert result["metadata_bytes"] == 64
        assert result["tensor_header_bytes"] == 32
        assert result["padding_bytes"] == 4
        assert result["total_bytes"] == 100 + 2 + 64 + 32 + 4


class TestValidateStorage:
    def test_int8_validation_pass(self):
        # Create test data
        tensor = np.random.default_rng(42).normal(0, 1, 1000).astype(np.float32)
        from src.ldmark.compression.quantize import quantize_int8, QuantizationConfig, QuantizationTarget
        
        config = QuantizationConfig(target_bits=QuantizationTarget.INT8, group_size=128, scale_dtype=np.float16)
        qtensor = quantize_int8(tensor, group_size=128)
        
        result = validate_storage(
            num_weights=tensor.size,
            bits_per_weight=8,
            group_size=128,
            quantized_data=qtensor.data,
            scales=qtensor.scales,
            scale_bits=16,
            tolerance=0.1,  # 10% tolerance
        )
        
        assert isinstance(result, StorageValidationResult)
        assert result.theoretical_bytes > 0
        assert result.actual_bytes > 0
        assert result.overhead_ratio >= 1.0  # Actual should be >= theoretical
        # Should pass with 10% tolerance
        assert result.validation_passed is True
    
    def test_int4_validation_pass(self):
        tensor = np.random.default_rng(42).normal(0, 1, 1000).astype(np.float32)
        from src.ldmark.compression.quantize import quantize_int4, QuantizationConfig, QuantizationTarget
        
        config = QuantizationConfig(target_bits=QuantizationTarget.INT4, group_size=128, scale_dtype=np.float16)
        qtensor = quantize_int4(tensor, group_size=128)
        
        result = validate_storage(
            num_weights=tensor.size,
            bits_per_weight=4,
            group_size=128,
            quantized_data=qtensor.data,
            scales=qtensor.scales,
            scale_bits=16,
            tolerance=0.1,
        )
        
        assert result.validation_passed is True
        assert result.overhead_ratio >= 1.0
    
    def test_binary_validation_pass(self):
        tensor = np.random.default_rng(42).normal(0, 1, 128).astype(np.float32)
        from src.ldmark.compression.binary_ternary import (
            binary_quantize_sign, BinaryConfig, calculate_binary_storage
        )
        
        config = BinaryConfig(group_size=128, scale_dtype=np.float16)
        btensor = binary_quantize_sign(tensor, config)
        
        result = validate_storage(
            num_weights=tensor.size,
            bits_per_weight=1,
            group_size=128,
            quantized_data=btensor.packed_data,
            scales=btensor.scales,
            scale_bits=16,
            tolerance=0.1,
        )
        
        assert result.validation_passed is True
        # For binary: 128 bits + 16 bits scale = 144 bits = 18 bytes theoretical
        # Actual: packed_data (16 bytes) + scales (2 bytes) = 18 bytes
        assert result.theoretical_bytes == 18
        assert result.actual_bytes == 18
    
    def test_storage_overhead_tracking(self):
        tensor = np.random.default_rng(42).normal(0, 1, 1000).astype(np.float32)
        from src.ldmark.compression.quantize import quantize_int8, QuantizationConfig, QuantizationTarget
        
        config = QuantizationConfig(target_bits=QuantizationTarget.INT8, group_size=128, scale_dtype=np.float16)
        qtensor = quantize_int8(tensor, group_size=128)
        
        result = validate_storage(
            num_weights=tensor.size,
            bits_per_weight=8,
            group_size=128,
            quantized_data=qtensor.data,
            scales=qtensor.scales,
            scale_bits=16,
            metadata_bytes=100,
            tensor_header_bytes=50,
            padding_bytes=10,
        )
        
        assert result.metadata_bytes == 100
        assert result.tensor_headers_bytes == 50
        assert result.padding_bytes == 10
        assert "data" in result.overhead_breakdown
        assert "scales" in result.overhead_breakdown
        assert "metadata" in result.overhead_breakdown
        assert "tensor_headers" in result.overhead_breakdown
        assert "padding" in result.overhead_breakdown


class TestValidateModelStorage:
    def test_model_storage_validation(self):
        from src.ldmark.benchmark.runner import run_tensor_benchmark
        from src.ldmark.benchmark.model import TensorBenchmarkResult, ErrorDistribution, TimingMetrics, StorageMetrics
        
        tensor = np.random.default_rng(42).normal(0, 1, (128, 128)).astype(np.float32)
        result1 = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test1",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        result2 = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test2",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=43,
        )
        
        tensor_results = [result1, result2]  # Simulate 2 tensors
        
        validation = validate_model_storage(tensor_results, tolerance=0.1)
        
        assert "model_validation_passed" in validation
        assert "total_theoretical_bytes" in validation
        assert "total_actual_bytes" in validation
        assert "total_overhead_ratio" in validation
        assert "per_tensor" in validation
        assert len(validation["per_tensor"]) == 2


class TestEstimateSerializedSize:
    def test_estimate_serialized_size(self):
        from src.ldmark.compression.quantize import quantize_int8, QuantizationConfig, QuantizationTarget
        
        tensors = {
            "layer1.weight": np.random.default_rng(1).normal(0, 1, (64, 128)).astype(np.float32),
            "layer1.bias": np.random.default_rng(2).normal(0, 1, (128,)).astype(np.float32),
        }
        
        quantized_tensors = {}
        scales = {}
        
        for name, tensor in tensors.items():
            config = QuantizationConfig(target_bits=QuantizationTarget.INT8, group_size=128, scale_dtype=np.float16)
            qtensor = quantize_int8(tensor, group_size=128)
            quantized_tensors[name] = qtensor
            scales[name] = qtensor.scales
        
        estimate = estimate_serialized_size(tensors, quantized_tensors, scales)
        
        assert "quantized_data_bytes" in estimate
        assert "scales_bytes" in estimate
        assert "format_headers_bytes" in estimate
        assert "estimated_total_bytes" in estimate
        assert estimate["estimated_total_bytes"] == (
            estimate["quantized_data_bytes"] + 
            estimate["scales_bytes"] + 
            estimate["format_headers_bytes"]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])