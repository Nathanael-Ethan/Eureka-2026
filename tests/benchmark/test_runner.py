"""
Tests for LDMARK Benchmark Laboratory - Runner
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from src.ldmark.benchmark.runner import (
    run_tensor_benchmark,
    run_model_benchmark,
    run_method_comparison,
    run_repeatability_benchmark,
    create_synthetic_model,
)
from src.ldmark.benchmark.model import ErrorDistribution, TimingMetrics, StorageMetrics


class TestRunTensorBenchmark:
    def test_fp32_benchmark(self):
        tensor = np.random.default_rng(42).normal(0, 1, (64, 64)).astype(np.float32)
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test_fp32",
            compression_method="FP32",
            target_bits=32,
            group_size=1,
            scale_dtype=np.float32,
            random_seed=42,
        )
        
        assert result.tensor_name == "test_fp32"
        assert result.compression_method == "FP32"
        assert result.target_bits == 32
        assert result.original_bytes == tensor.nbytes
        assert result.compressed_bytes == tensor.nbytes  # No compression
        assert result.compression_ratio == 1.0
        assert result.actual_bits_per_weight == 32.0
        assert result.error_distribution.mean == 0.0  # Perfect reconstruction
        assert result.relative_error == 0.0
    
    def test_fp16_benchmark(self):
        tensor = np.random.default_rng(42).normal(0, 1, (64, 64)).astype(np.float32)
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test_fp16",
            compression_method="FP16",
            target_bits=16,
            group_size=1,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert result.compression_method == "FP16"
        assert result.target_bits == 16
        assert result.compressed_bytes == tensor.nbytes // 2  # FP16 is half size
        assert result.compression_ratio == 2.0
        assert result.actual_bits_per_weight == 16.0
        assert result.error_distribution.mean > 0  # Some error from FP32->FP16
    
    def test_int8_benchmark(self):
        tensor = np.random.default_rng(42).normal(0, 1, (128, 128)).astype(np.float32)
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test_int8",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert result.compression_method == "INT8"
        assert result.target_bits == 8
        assert result.group_size == 128
        assert result.compression_ratio > 3.0  # Should be ~4x
        assert result.compression_ratio < 5.0
        assert result.actual_bits_per_weight > 8.0  # With scale overhead
        assert result.actual_bits_per_weight < 9.0
        assert result.error_distribution.mean > 0
        assert result.storage.scales_bytes > 0
    
    def test_int4_benchmark(self):
        tensor = np.random.default_rng(42).normal(0, 1, (128, 128)).astype(np.float32)
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test_int4",
            compression_method="INT4",
            target_bits=4,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert result.compression_method == "INT4"
        assert result.target_bits == 4
        assert result.compression_ratio > 6.0  # Should be ~8x
        assert result.compression_ratio < 10.0
        assert result.actual_bits_per_weight > 4.0  # With scale overhead
        assert result.actual_bits_per_weight < 5.0
        assert result.storage.scales_bytes > 0
    
    def test_binary_benchmark(self):
        tensor = np.random.default_rng(42).normal(0, 1, (128, 128)).astype(np.float32)
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test_binary",
            compression_method="BINARY",
            target_bits=1,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert result.compression_method == "BINARY"
        assert result.target_bits == 1
        assert result.actual_bits_per_weight > 1.0  # With scale overhead
        assert result.actual_bits_per_weight < 1.2
        assert result.storage.scales_bytes > 0
    
    def test_ternary_benchmark(self):
        tensor = np.random.default_rng(42).normal(0, 1, (128, 128)).astype(np.float32)
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name="test_ternary",
            compression_method="TERNARY",
            target_bits=int(np.log2(3) * 1000) / 1000,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert result.compression_method == "TERNARY"
        assert result.target_bits > 1.5
        assert result.target_bits < 1.6
        assert result.storage.scales_bytes > 0
    
    def test_invalid_method(self):
        tensor = np.ones((10,), dtype=np.float32)
        with pytest.raises(ValueError):
            run_tensor_benchmark(
                tensor=tensor,
                tensor_name="test",
                compression_method="INVALID",
                target_bits=8,
                group_size=128,
                scale_dtype=np.float16,
                random_seed=42,
            )


class TestRunModelBenchmark:
    def test_model_benchmark(self):
        tensors = {
            "layer1.weight": np.random.default_rng(1).normal(0, 1, (64, 128)).astype(np.float32),
            "layer1.bias": np.random.default_rng(2).normal(0, 1, (128,)).astype(np.float32),
            "layer2.weight": np.random.default_rng(3).normal(0, 1, (128, 10)).astype(np.float32),
            "layer2.bias": np.random.default_rng(4).normal(0, 1, (10,)).astype(np.float32),
        }
        
        result = run_model_benchmark(
            tensors=tensors,
            model_identifier="test_model",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert result.model_identifier == "test_model"
        assert result.tensor_count == 4
        assert result.parameter_count == sum(t.size for t in tensors.values())
        assert len(result.tensor_results) == 4
        assert result.aggregate_compression_ratio > 3.0
        assert result.aggregate_bits_per_weight > 8.0


class TestRunMethodComparison:
    def test_method_comparison(self):
        tensor = np.random.default_rng(42).normal(0, 1, (128, 128)).astype(np.float32)
        methods = [
            ("FP32", 32, 1, np.float32),
            ("FP16", 16, 1, np.float16),
            ("INT8", 8, 128, np.float16),
            ("INT4", 4, 128, np.float16),
        ]
        
        results = run_method_comparison(
            tensor=tensor,
            tensor_name="comparison_test",
            methods=methods,
            random_seed=42,
        )
        
        assert len(results) == 4
        assert results[0].compression_method == "FP32"
        assert results[1].compression_method == "FP16"
        assert results[2].compression_method == "INT8"
        assert results[3].compression_method == "INT4"
        
        # Verify compression ratios increase
        assert results[0].compression_ratio == 1.0
        assert results[1].compression_ratio == 2.0
        assert results[2].compression_ratio > results[1].compression_ratio
        assert results[3].compression_ratio > results[2].compression_ratio


class TestRunRepeatabilityBenchmark:
    def test_repeatability_deterministic(self):
        tensor = np.random.default_rng(42).normal(0, 1, (256, 256)).astype(np.float32)
        
        results, analysis = run_repeatability_benchmark(
            tensor=tensor,
            tensor_name="repeat_test",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype=np.float16,
            num_runs=5,
            base_seed=42,
        )
        
        assert len(results) == 5
        assert analysis["num_runs"] == 5
        assert analysis["all_results_identical"] is True
        assert analysis["bits_per_weight"]["std"] == 0.0
        assert len(analysis["seeds_used"]) == 5
    
    def test_repeatability_different_seeds(self):
        tensor = np.random.default_rng(42).normal(0, 1, (256, 256)).astype(np.float32)
        
        results, analysis = run_repeatability_benchmark(
            tensor=tensor,
            tensor_name="repeat_test",
            compression_method="INT8",
            target_bits=8,
            group_size=128,
            scale_dtype=np.float16,
            num_runs=3,
            base_seed=100,
        )
        
        assert analysis["num_runs"] == 3
        # With different seeds, tensor generation differs but quantization is deterministic
        # So the quantization results should be identical for the same input tensor
        # But since we use different seeds for tensor generation, inputs differ
        # The key point: same seed = same result


class TestCreateSyntheticModel:
    def test_create_synthetic_model(self):
        specs = [
            ("weight1", (64, 128), "float32"),
            ("bias1", (128,), "float32"),
            ("weight2", (128, 10), "float16"),
        ]
        
        tensors = create_synthetic_model(specs, generator="random_normal", seed=42)
        
        assert len(tensors) == 3
        assert "weight1" in tensors
        assert "bias1" in tensors
        assert "weight2" in tensors
        assert tensors["weight1"].shape == (64, 128)
        assert tensors["weight1"].dtype == np.float32
        assert tensors["weight2"].dtype == np.float16
    
    def test_different_generators(self):
        specs = [("test", (100,), "float32")]
        
        tensors_normal = create_synthetic_model(specs, generator="random_normal", seed=42)
        tensors_uniform = create_synthetic_model(specs, generator="random_uniform", seed=42)
        tensors_ones = create_synthetic_model(specs, generator="ones", seed=42)
        tensors_zeros = create_synthetic_model(specs, generator="zeros", seed=42)
        
        assert np.all(tensors_ones["test"] == 1.0)
        assert np.all(tensors_zeros["test"] == 0.0)
        assert np.std(tensors_normal["test"]) > 0
        assert np.std(tensors_uniform["test"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])