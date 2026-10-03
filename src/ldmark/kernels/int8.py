"""
LDMARK INT8 Direct Computation Kernel

Group-wise INT8 matrix multiplication without full dequantization.
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
    
    Supports direct matrix multiplication using the quantized representation.
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
                        If None, assumes original_shape is (out_features, in_features)
        """
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
        
        # Pre-process scales for efficient computation
        self._scales_fp32 = quantized_tensor.scales.astype(np.float32)
        self._data = quantized_tensor.data
        
        # Compute size info
        self._compressed_bytes = quantized_tensor.data.nbytes + quantized_tensor.scales.nbytes
        self._decompressed_bytes = self._rows * self._cols * 4  # FP32
    
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
        """
        Direct INT8 GEMM: Y = W @ X
        
        Uses group-wise scaling: each group of weights has its own scale.
        Computation: Y = sum_g (scale_g * (Q_g @ X_g))
        
        Args:
            x: Input matrix of shape (K, N) or vector (K,)
            config: Computation configuration
            
        Returns:
            ComputeResult with output and statistics
        """
        return Int8Matmul().compute(self, x, config)


class Int8Matmul(LowBitMatmul):
    """INT8 direct matrix multiplication kernel."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "int8_direct"
    
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
        Compute Y = W @ X using group-wise INT8 representation.
        
        Algorithm:
        1. For each group g:
           - Extract weight slice Q_g (group_size, K)
           - Extract input slice X_g (group_size, N)  
           - Compute Y_g = Q_g @ X_g  (in int32 to avoid overflow)
           - Accumulate: Y += scale_g * Y_g
        
        This avoids materializing the full dequantized W matrix.
        """
        # Handle both vector and matrix input
        x_is_vector = x.ndim == 1
        if x_is_vector:
            x = x.reshape(-1, 1)
        
        K, N = x.shape
        assert K == weight._cols, f"Input dim {K} != weight cols {weight._cols}"
        
        start_prep = time.perf_counter()
        
        # Prepare output
        output_dtype = np.float32 if config.output_precision == ComputePrecision.FP32 else np.float16
        y = np.zeros((weight._rows, N), dtype=output_dtype)
        
        # Track memory
        temp_arrays = []
        
        prep_time = time.perf_counter() - start_prep
        start_compute = time.perf_counter()
        
        # Process each group
        group_size = weight._group_size
        scales = weight._scales_fp32
        data = weight._data
        
        n_elements = weight._rows * weight._cols
        n_groups = len(scales)
        
        # We need to iterate over output rows (weight rows)
        # For each output row, it spans multiple groups across K dimension
        
        # Approach: iterate over groups along K dimension
        # For group g, we have:
        # - weight slice: rows [0, weight._rows), cols [g*group_size, min((g+1)*group_size, K)]
        # - input slice: rows [g*group_size, min((g+1)*group_size, K)], cols [0, N]
        
        for g in range(n_groups):
            scale = scales[g]
            if scale == 0:
                continue
                
            k_start = g * group_size
            k_end = min(k_start + group_size, K)
            group_k = k_end - k_start
            
            # Extract weight slice for this group: shape (out_features, group_k)
            # The quantized data is stored in row-major order, grouped by K dimension
            elements_before = weight._rows * k_start
            if weight._qtensor.config.target_bits == QuantizationTarget.INT8:
                bytes_before = elements_before
                group_len = weight._rows * group_k
                w_group = data[bytes_before:bytes_before + group_len].reshape(weight._rows, group_k)
            else:
                raise ValueError("Int8Matrix expects INT8 quantized tensor")
            
            # Extract input slice: shape (group_k, N)
            x_group = x[k_start:k_end, :]
            
            # Compute partial result: (out_features, group_k) @ (group_k, N) -> (out_features, N)
            # Use int32 accumulation to avoid overflow
            partial = w_group.astype(np.int32) @ x_group.astype(np.int32)
            
            # Accumulate with scale
            if config.accumulate_in_fp32:
                y += (partial.astype(np.float32) * scale).astype(output_dtype)
            else:
                y += (partial * scale).astype(output_dtype)
        
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
                temporary_bytes=0,  # No large temporaries created
                peak_bytes=weight.compressed_size_bytes + x.nbytes + y.nbytes,
            ),
            timing_stats=TimingStats(
                preparation_time_s=prep_time,
                computation_time_s=compute_time,
                total_time_s=prep_time + compute_time,
            ),
            metadata={
                "method": "int8_direct_groupwise",
                "group_size": group_size,
                "num_groups": n_groups,
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
    """Direct INT8 matmul."""
    return Int8Matmul().compute(weight, x, config)


def benchmark_int8(
    weight: Int8Matrix,
    x: np.ndarray,
    config: ComputationConfig,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark INT8 direct vs reference."""
    # Warmup
    for _ in range(3):
        _ = int8_matmul_direct(weight, x, config)
        _ = int8_matmul_reference(weight, x, config)
    
    # Benchmark direct
    direct_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        direct = int8_matmul_direct(weight, x, config)
        direct_times.append(time.perf_counter() - start)
    
    # Benchmark reference
    ref_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ref = int8_matmul_reference(weight, x, config)
        ref_times.append(time.perf_counter() - start)
    
    direct = int8_matmul_direct(weight, x, config)
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