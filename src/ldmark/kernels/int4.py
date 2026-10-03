"""
LDMARK INT4 Direct Computation Kernel

Packed INT4 matrix multiplication with group-wise scaling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time

from ldmark.compute.abstraction import (
    LowBitMatrix,
    LowBitMatmul,
    ComputationConfig,
    ComputeResult,
    MemoryStats,
    TimingStats,
    ComputePrecision,
    create_reference_result,
    compare_results,
)
from ldmark.compression.quantize import (
    QuantizedTensor,
    QuantizationConfig,
    QuantizationTarget,
    dequantize_groupwise,
    _unpack_int4,
)


class Int4Matrix(LowBitMatrix):
    """
    Packed INT4 quantized matrix with group-wise scaling.
    
    Weights stored as packed uint8 (2 INT4 values per byte).
    Supports direct matrix multiplication on packed representation.
    """
    
    def __init__(
        self,
        quantized_tensor: QuantizedTensor,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize from a QuantizedTensor.
        
        Args:
            quantized_tensor: The quantized weight tensor (must be INT4)
            input_shape: Optional explicit input shape (out_features, in_features)
        """
        if quantized_tensor.config.target_bits != QuantizationTarget.INT4:
            raise ValueError("Int4Matrix expects INT4 quantized tensor")
            
        self._qtensor = quantized_tensor
        
        # Determine matrix shape
        if input_shape is not None:
            self._shape = input_shape
        else:
            orig_shape = quantized_tensor.original_shape
            if len(orig_shape) == 2:
                self._shape = orig_shape
            else:
                raise ValueError(f"Cannot infer matrix shape from {orig_shape}. Provide input_shape.")
        
        self._rows, self._cols = self._shape
        self._group_size = quantized_tensor.config.group_size
        self._symmetric = quantized_tensor.config.symmetric
        
        # Pre-process scales
        self._scales_fp32 = quantized_tensor.scales.astype(np.float32)
        self._packed_data = quantized_tensor.data
        
        # Compute size info
        self._compressed_bytes = quantized_tensor.data.nbytes + quantized_tensor.scales.nbytes
        self._decompressed_bytes = self._rows * self._cols * 4  # FP32
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)  # Packed storage
    
    @property
    def compressed_size_bytes(self) -> int:
        return self._compressed_bytes
    
    @property
    def decompressed_size_bytes(self) -> int:
        return self._decompressed_bytes
    
    def dequantize(self) -> np.ndarray:
        """Full dequantization to FP32 for reference."""
        return dequantize_groupwise(self._qtensor).astype(np.float32)
    
    def matmul(self, x: np.ndarray, config: ComputationConfig) -> ComputeResult:
        """Direct INT4 GEMM on packed representation."""
        return Int4Matmul().compute(self, x, config)


class Int4Matmul(LowBitMatmul):
    """INT4 direct matrix multiplication kernel on packed data."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "int4_direct_packed"
    
    @property
    def supported_precisions(self) -> List[ComputePrecision]:
        return [ComputePrecision.FP32, ComputePrecision.FP16]
    
    def compute(
        self,
        weight: Int4Matrix,
        x: np.ndarray,
        config: ComputationConfig
    ) -> ComputeResult:
        """
        Compute Y = W @ X using packed INT4 representation.
        
        For INT4, weights are packed as 2 nibbles per uint8.
        We unpack on-the-fly during computation to avoid full dequantization.
        
        Algorithm:
        1. For each group along K dimension:
           - Unpack weight slice for this group
           - Extract input slice
           - Compute partial: unpacked_Q_g @ X_g
           - Accumulate: Y += scale_g * partial
        """
        # Handle both vector and matrix input
        x_is_vector = x.ndim == 1
        if x_is_vector:
            x = x.reshape(-1, 1)
        
        K, N = x.shape
        assert K == weight._cols, f"Input dim {K} != weight cols {weight._cols}"
        
        start_prep = time.perf_counter()
        
        output_dtype = np.float32 if config.output_precision == ComputePrecision.FP32 else np.float16
        y = np.zeros((weight._rows, N), dtype=output_dtype)
        
        prep_time = time.perf_counter() - start_prep
        start_compute = time.perf_counter()
        
        group_size = weight._group_size
        scales = weight._scales_fp32
        packed = weight._packed_data
        
        n_groups = len(scales)
        
        # Process each group along K dimension
        for g in range(n_groups):
            scale = scales[g]
            if scale == 0:
                continue
                
            k_start = g * group_size
            k_end = min(k_start + group_size, K)
            group_k = k_end - k_start
            
            # Calculate packed data indices for this group
            # Each weight row has ceil(K/2) bytes
            bytes_per_row = (K + 1) // 2
            
            # For this group, we need group_k columns
            # Each row contributes group_k nibbles = ceil(group_k/2) bytes
            group_bytes_per_row = (group_k + 1) // 2
            
            # Unpack weight slice for this group: (rows, group_k)
            w_group = np.zeros((weight._rows, group_k), dtype=np.int8)
            
            for r in range(weight._rows):
                row_start_byte = r * bytes_per_row + (k_start // 2)
                group_packed = packed[row_start_byte:row_start_byte + group_bytes_per_row]
                w_group[r, :] = _unpack_int4(group_packed, group_k)
            
            # Extract input slice: shape (group_k, N)
            x_group = x[k_start:k_end, :]
            
            # Compute partial: (rows, group_k) @ (group_k, N)
            # Accumulate in int32 to avoid overflow
            partial = w_group.astype(np.int32) @ x_group.astype(np.int32)
            
            # Accumulate with scale
            if config.accumulate_in_fp32:
                y += (partial.astype(np.float32) * scale).astype(y.dtype)
            else:
                y += (partial * scale).astype(y.dtype)
        
        compute_time = time.perf_counter() - start_compute
        
        if x_is_vector:
            y = y.flatten()
        
        return ComputeResult(
            output=y,
            memory_stats=MemoryStats(
                compressed_weight_bytes=weight.compressed_size_bytes,
                decompressed_weight_bytes=weight.decompressed_size_bytes,
                input_bytes=x.nbytes,
                output_bytes=y.nbytes,
                temporary_bytes=weight._rows * group_k * 1,  # unpacked group buffer
                peak_bytes=weight.compressed_size_bytes + x.nbytes + y.nbytes,
            ),
            timing_stats=TimingStats(
                preparation_time_s=prep_time,
                computation_time_s=compute_time,
                total_time_s=prep_time + compute_time,
            ),
            metadata={
                "method": "int4_direct_packed",
                "group_size": group_size,
                "num_groups": n_groups,
            }
        )


def int4_matmul_reference(
    weight: Int4Matrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Reference: dequantize then matmul."""
    return create_reference_result(weight, x, config)


def int4_matmul_direct(
    weight: Int4Matrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Direct INT4 matmul."""
    return Int4Matmul().compute(weight, x, config)


def benchmark_int4(
    weight: Int4Matrix,
    x: np.ndarray,
    config: ComputationConfig,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark INT4 direct vs reference."""
    # Warmup
    for _ in range(3):
        _ = int4_matmul_direct(weight, x, config)
        _ = int4_matmul_reference(weight, x, config)
    
    # Benchmark direct
    direct_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        direct = int4_matmul_direct(weight, x, config)
        direct_times.append(time.perf_counter() - start)
    
    # Benchmark reference
    ref_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ref = int4_matmul_reference(weight, x, config)
        ref_times.append(time.perf_counter() - start)
    
    direct = int4_matmul_direct(weight, x, config)
    ref = int4_matmul_reference(weight, x, config)
    comparison = compare_results(direct, ref)
    
    return {
        "direct": {
            "mean_time_s": float(np.mean(direct_times)),
            "std_time_s": float(np.std(direct_times)),
            "min_time_s": float(np.min(direct_times)),
            "max_time_s": float(np.max(direct_times)),
        },
        "reference": {
            "mean_time_s": float(np.mean(ref_times)),
            "std_time_s": float(np.std(ref_times)),
            "min_time_s": float(np.min(ref_times)),
            "max_time_s": float(np.max(ref_times)),
        },
        "comparison": comparison,
        "memory": {
            "compressed_bytes": weight.compressed_size_bytes,
            "decompressed_bytes": weight.decompressed_size_bytes,
        }
    }