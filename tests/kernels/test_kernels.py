"""
Tests for LDMARK Low-Bit Computation Kernels
"""

from __future__ import annotations

import numpy as np
import pytest

from ldmark.compute import (
    ComputationConfig,
    ComputePrecision,
    create_reference_result,
    compare_results,
)
from ldmark.kernels import (
    Int8Matrix,
    int8_matmul_direct,
    int8_matmul_reference,
    Int4Matrix,
    int4_matmul_direct,
    int4_matmul_reference,
    BinaryMatrix,
    binary_matmul_direct,
    binary_matmul_reference,
    TernaryMatrix,
    ternary_matmul_direct,
    ternary_matmul_reference,
)
from ldmark.compression.quantize import (
    quantize_int8,
    quantize_int4,
    QuantizationConfig,
    QuantizationTarget,
)
from ldmark.compression.binary_ternary import (
    binary_quantize_sign,
    ternary_quantize_threshold,
    BinaryConfig,
    TernaryConfig,
)


def create_test_matrix(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
    """Create a test matrix with controlled values."""
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


class TestInt8Computation:
    """Tests for INT8 direct computation."""
    
    def test_int8_square_matrix(self):
        """Test INT8 on square matrix."""
        weight_fp32 = create_test_matrix((64, 64))
        x = create_test_matrix((64, 32))
        
        qconfig = QuantizationConfig(target_bits=QuantizationTarget.INT8, group_size=32)
        qtensor = quantize_int8(weight_fp32, group_size=32)
        
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig(
            output_precision=ComputePrecision.FP32,
            accumulate_in_fp32=True,
        )
        
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"], f"INT8 failed: {comparison}"
    
    def test_int8_rectangular_matrix(self):
        """Test INT8 on rectangular matrix."""
        weight_fp32 = create_test_matrix((128, 64))
        x = create_test_matrix((64, 16))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int8_vector_input(self):
        """Test INT8 with vector input."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32,))  # Vector
        
        qtensor = quantize_int8(weight_fp32, group_size=16)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        assert direct.output.shape == (32,)
        assert reference.output.shape == (32,)
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int8_odd_dimensions(self):
        """Test INT8 with odd dimensions."""
        weight_fp32 = create_test_matrix((33, 31))
        x = create_test_matrix((31, 7))
        
        qtensor = quantize_int8(weight_fp32, group_size=16)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int8_non_divisible_group(self):
        """Test INT8 with non-divisible group size."""
        weight_fp32 = create_test_matrix((50, 50))
        x = create_test_matrix((50, 10))
        
        qtensor = quantize_int8(weight_fp32, group_size=17)  # 50 % 17 != 0
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int8_single_row(self):
        """Test INT8 with single row weight."""
        weight_fp32 = create_test_matrix((1, 64))
        x = create_test_matrix((64, 8))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        assert direct.output.shape == (1, 8)
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int8_single_column(self):
        """Test INT8 with single column weight."""
        weight_fp32 = create_test_matrix((64, 1))
        x = create_test_matrix((1, 8))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int8_matmul_direct(weight, x, config)
        reference = int8_matmul_reference(weight, x, config)
        
        assert direct.output.shape == (64, 8)
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]


class TestInt4Computation:
    """Tests for INT4 direct computation."""
    
    def test_int4_square_matrix(self):
        """Test INT4 on square matrix."""
        weight_fp32 = create_test_matrix((64, 64))
        x = create_test_matrix((64, 32))
        
        qconfig = QuantizationConfig(target_bits=QuantizationTarget.INT4, group_size=32)
        qtensor = quantize_int4(weight_fp32, group_size=32)
        
        weight = Int4Matrix(qtensor)
        
        config = ComputationConfig(
            output_precision=ComputePrecision.FP32,
            accumulate_in_fp32=True,
        )
        
        direct = int4_matmul_direct(weight, x, config)
        reference = int4_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"], f"INT4 failed: {comparison}"
        assert comparison["max_absolute_error"] < 1e-3
    
    def test_int4_rectangular_matrix(self):
        """Test INT4 on rectangular matrix."""
        weight_fp32 = create_test_matrix((64, 128))
        x = create_test_matrix((128, 16))
        
        qtensor = quantize_int4(weight_fp32, group_size=32)
        weight = Int4Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int4_matmul_direct(weight, x, config)
        reference = int4_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int4_vector_input(self):
        """Test INT4 with vector input."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32,))
        
        qtensor = quantize_int4(weight_fp32, group_size=16)
        weight = Int4Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int4_matmul_direct(weight, x, config)
        reference = int4_matmul_reference(weight, x, config)
        
        assert direct.output.shape == (32,)
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int4_odd_dimensions(self):
        """Test INT4 with odd dimensions."""
        weight_fp32 = create_test_matrix((33, 31))
        x = create_test_matrix((31, 7))
        
        qtensor = quantize_int4(weight_fp32, group_size=16)
        weight = Int4Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int4_matmul_direct(weight, x, config)
        reference = int4_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_int4_non_divisible_group(self):
        """Test INT4 with non-divisible group size."""
        weight_fp32 = create_test_matrix((50, 50))
        x = create_test_matrix((50, 10))
        
        qtensor = quantize_int4(weight_fp32, group_size=17)
        weight = Int4Matrix(qtensor)
        
        config = ComputationConfig()
        direct = int4_matmul_direct(weight, x, config)
        reference = int4_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]


class TestBinaryComputation:
    """Tests for Binary direct computation."""
    
    def test_binary_square_matrix(self):
        """Test Binary on square matrix."""
        weight_fp32 = create_test_matrix((64, 64))
        x = create_test_matrix((64, 32))
        
        bconfig = BinaryConfig(encoding="sign", group_size=32)
        btensor = binary_quantize_sign(weight_fp32, bconfig)
        
        weight = BinaryMatrix(btensor)
        
        config = ComputationConfig(
            output_precision=ComputePrecision.FP32,
            accumulate_in_fp32=True,
        )
        
        direct = binary_matmul_direct(weight, x, config)
        reference = binary_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"], f"Binary failed: {comparison}"
    
    def test_binary_rectangular_matrix(self):
        """Test Binary on rectangular matrix."""
        weight_fp32 = create_test_matrix((32, 128))
        x = create_test_matrix((128, 16))
        
        btensor = binary_quantize_sign(weight_fp32, BinaryConfig(group_size=32))
        weight = BinaryMatrix(btensor)
        
        config = ComputationConfig()
        direct = binary_matmul_direct(weight, x, config)
        reference = binary_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_binary_vector_input(self):
        """Test Binary with vector input."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32,))
        
        btensor = binary_quantize_sign(weight_fp32, BinaryConfig(group_size=16))
        weight = BinaryMatrix(btensor)
        
        config = ComputationConfig()
        direct = binary_matmul_direct(weight, x, config)
        reference = binary_matmul_reference(weight, x, config)
        
        assert direct.output.shape == (32,)
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]


class TestTernaryComputation:
    """Tests for Ternary direct computation."""
    
    def test_ternary_square_matrix(self):
        """Test Ternary on square matrix."""
        weight_fp32 = create_test_matrix((64, 64))
        x = create_test_matrix((64, 32))
        
        tconfig = TernaryConfig(encoding="minus_zero_plus", group_size=32)
        ttensor = ternary_quantize_threshold(weight_fp32, tconfig, threshold=0.05)
        
        weight = TernaryMatrix(ttensor)
        
        config = ComputationConfig(
            output_precision=ComputePrecision.FP32,
            accumulate_in_fp32=True,
        )
        
        direct = ternary_matmul_direct(weight, x, config)
        reference = ternary_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"], f"Ternary failed: {comparison}"
    
    def test_ternary_rectangular_matrix(self):
        """Test Ternary on rectangular matrix."""
        weight_fp32 = create_test_matrix((32, 128))
        x = create_test_matrix((128, 16))
        
        ttensor = ternary_quantize_threshold(weight_fp32, TernaryConfig(group_size=32))
        weight = TernaryMatrix(ttensor)
        
        config = ComputationConfig()
        direct = ternary_matmul_direct(weight, x, config)
        reference = ternary_matmul_reference(weight, x, config)
        
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]
    
    def test_ternary_vector_input(self):
        """Test Ternary with vector input."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32,))
        
        ttensor = ternary_quantize_threshold(weight_fp32, TernaryConfig(group_size=16))
        weight = TernaryMatrix(ttensor)
        
        config = ComputationConfig()
        direct = ternary_matmul_direct(weight, x, config)
        reference = ternary_matmul_reference(weight, x, config)
        
        assert direct.output.shape == (32,)
        comparison = compare_results(direct, reference, rtol=1e-4, atol=1e-5)
        assert comparison["passed"]


class TestComputationConfig:
    """Tests for ComputationConfig options."""
    
    def test_fp16_output(self):
        """Test FP16 output precision."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32, 16))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig(output_precision=ComputePrecision.FP16)
        direct = int8_matmul_direct(weight, x, config)
        
        assert direct.output.dtype == np.float16
    
    def test_fp32_accumulation(self):
        """Test FP32 accumulation vs non-FP32."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32, 8))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        
        config_fp32 = ComputationConfig(accumulate_in_fp32=True)
        config_no_fp32 = ComputationConfig(accumulate_in_fp32=False)
        
        direct_fp32 = int8_matmul_direct(weight, x, config_fp32)
        direct_no_fp32 = int8_matmul_direct(weight, x, config_no_fp32)
        
        # Results should be very close (both should be accurate)
        diff = np.abs(direct_fp32.output - direct_no_fp32.output)
        assert np.max(diff) < 1e-4


class TestMemoryStats:
    """Tests for memory statistics tracking."""
    
    def test_memory_stats_populated(self):
        """Test that memory stats are populated."""
        weight_fp32 = create_test_matrix((64, 64))
        x = create_test_matrix((64, 32))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        
        config = ComputationConfig(enable_memory_tracking=True)
        result = int8_matmul_direct(weight, x, config)
        
        assert result.memory_stats.compressed_weight_bytes > 0
        assert result.memory_stats.decompressed_weight_bytes > 0
        assert result.memory_stats.input_bytes == x.nbytes
        assert result.memory_stats.output_bytes == result.output.nbytes
    
    def test_compressed_vs_decompressed(self):
        """Test compressed size < decompressed size."""
        weight_fp32 = create_test_matrix((128, 128))
        x = create_test_matrix((128, 64))
        
        for qfunc, MatrixClass in [
            (quantize_int8, Int8Matrix),
            (quantize_int4, Int4Matrix),
        ]:
            qtensor = qfunc(weight_fp32, group_size=32)
            weight = MatrixClass(qtensor)
            
            assert weight.compressed_size_bytes < weight.decompressed_size_bytes
            assert weight.compressed_size_bytes > 0


class TestDeterminism:
    """Tests for deterministic results."""
    
    def test_deterministic_int8(self):
        """INT8 results should be deterministic."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32, 16))
        
        qtensor = quantize_int8(weight_fp32, group_size=32)
        weight = Int8Matrix(qtensor)
        config = ComputationConfig()
        
        r1 = int8_matmul_direct(weight, x, config)
        r2 = int8_matmul_direct(weight, x, config)
        
        assert np.allclose(r1.output, r2.output)
        assert r1.timing_stats.computation_time_s >= 0
    
    def test_deterministic_int4(self):
        """INT4 results should be deterministic."""
        weight_fp32 = create_test_matrix((32, 32))
        x = create_test_matrix((32, 16))
        
        qtensor = quantize_int4(weight_fp32, group_size=32)
        weight = Int4Matrix(qtensor)
        config = ComputationConfig()
        
        r1 = int4_matmul_direct(weight, x, config)
        r2 = int4_matmul_direct(weight, x, config)
        
        assert np.allclose(r1.output, r2.output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])