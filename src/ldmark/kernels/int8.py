"""
LDMARK INT8 Direct Computation Kernel

Group-wise INT8 matrix multiplication without full dequantization.
Uses horizontal group layout (per-row) for the common case.
Falls back to dequantization for edge cases where groups don't align with rows.
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
)


class Int8Matrix(LowBitMatrix):
    """
    INT8 quantized matrix with group-wise scaling.
    
    Supports direct matrix multiplication using the quantized representation
    when groups align with rows (horizontal grouping).
    """
    
    def __init__(
        self,
        quantized_tensor: QuantizedTensor,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize from a QuantizedTensor.
        
        Args:
            quantized_tensor: The quantized weight tensor
            input_shape: Optional explicit input shape (out_features, in_features)
        """
        self._qtensor = quantized_tensor
        
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
        
        self._scales_fp32 = quantized_tensor.scales.astype(np.float32)
        self._data = quantized_tensor.data
        self._n_groups = len(quantized_tensor.scales)
        
        self._compressed_bytes = quantized_tensor.data.nbytes + quantized_tensor.scales.nbytes
        self._decompressed_bytes = self._rows * self._cols * 4
        
        # Check if horizontal grouping is valid
        self._groups_per_row = self._cols // self._group_size
        if self._cols % self._group_size != 0:
            self._groups_per_row += 1
        self._horizontal_valid = (self._cols >= self._group_size and 
                                   self._cols % self._group_size == 0 and
                                   self._n_groups == self._rows * (self._cols // self._group_size))
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.int8)
    
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
        """Direct INT8 GEMM: Y = W @ X"""
        if self._horizontal_valid:
            return Int8MatmulHorizontal().compute(self, x, config)
        else:
            # Fallback to dequantize-then-matmul for edge cases
            return self._fallback_matmul(x, config)
    
    def _fallback_matmul(self, x: np.ndarray, config: ComputationConfig) -> ComputeResult:
        """Fallback: dequantize then matmul."""
        from ldmark.compute.abstraction import create_reference_result
        return create_reference_result(self, x, config)


class Int8MatmulHorizontal(LowBitMatmul):
    """INT8 direct matmul for horizontal group layout (groups per row)."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "int8_direct_horizontal"
    
    @property
    def supported_precisions(self) -> List[ComputePrecision]:
        return [ComputePrecision.FP32, ComputePrecision.FP16]
    
    def compute(
        self,
        weight: Int8Matrix,
        x: np.ndarray,
        config: ComputationConfig
    ) -> ComputeResult:
        """
        Compute Y = W @ X using horizontal group layout.
        
        Assumes groups are horizontal slices per row: each row has groups_per_row groups
        covering contiguous chunks of columns.
        """
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
        data = weight._data
        groups_per_row = weight._groups_per_row
        n_groups = len(scales)
        
        # Iterate over output rows
        for r in range(weight._rows):
            row_start = r * weight._cols
            
            for g in range(groups_per_row):
                group_idx = r * groups_per_row + g
                if group_idx >= n_groups:
                    break
                    
                scale = scales[group_idx]
                if scale == 0:
                    continue
                
                k_start = g * group_size
                k_end = min(k_start + group_size, K)
                group_k = k_end - k_start
                
                # Weight slice for this row's group
                elements_before = row_start + g * group_size
                w_group = data[elements_before:elements_before + group_k]
                
                # Input slice
                x_group = x[k_start:k_end, :]
                
                # Compute partial: (1, group_k) @ (group_k, N)
                partial = w_group.astype(np.int32) @ x_group.astype(np.float32)
                
                # Accumulate
                if config.accumulate_in_fp32:
                    y[r, :] += (partial * scale).astype(output_dtype)
                else:
                    y[r, :] += (partial * scale).astype(output_dtype)
        
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
                temporary_bytes=0,
                peak_bytes=weight.compressed_size_bytes + x.nbytes + y.nbytes,
            ),
            timing_stats=TimingStats(
                preparation_time_s=prep_time,
                computation_time_s=compute_time,
                total_time_s=prep_time + compute_time,
            ),
            metadata={
                "method": "int8_direct_horizontal",
                "group_size": group_size,
                "num_groups": n_groups,
                "horizontal": True,
            }
        )


def int8_matmul_reference(
    weight: Int8Matrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Reference: dequantize then matmul."""
    return create_reference_result(weight, x, config)


def int8_matmul_direct(
    weight: Int8Matrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Direct INT8 matmul (uses horizontal or fallback)."""
    return weight.matmul(x, config)


def benchmark_int8(
    weight: Int8Matrix,
    x: np.ndarray,
    config: ComputationConfig,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark INT8 direct vs reference."""
    for _ in range(3):
        _ = weight.matmul(x, config)
        _ = int8_matmul_reference(weight, x, config)
    
    direct_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        direct = weight.matmul(x, config)
        direct_times.append(time.perf_counter() - start)
    
    ref_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ref = int8_matmul_reference(weight, x, config)
        ref_times.append(time.perf_counter() - start)
    
    direct = weight.matmul(x, config)
    ref = int8_matmul_reference(weight, x, config)
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