"""
LDMARK Binary Direct Computation Kernel

Binary sign weight matrix multiplication using packed bits.
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
from ldmark.compression.binary_ternary import (
    BinaryTensor,
    BinaryConfig,
    BinaryEncoding,
    binary_dequantize,
)


class BinaryMatrix(LowBitMatrix):
    """
    Binary quantized matrix (sign weights {-1, +1}).
    
    Weights stored as packed bits (1 bit per weight).
    """
    
    def __init__(
        self,
        binary_tensor: BinaryTensor,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize from a BinaryTensor.
        
        Args:
            binary_tensor: The binary weight tensor
            input_shape: Optional explicit input shape (out_features, in_features)
        """
        self._btensor = binary_tensor
        
        # Determine matrix shape
        if input_shape is not None:
            self._shape = input_shape
        else:
            orig_shape = binary_tensor.original_shape
            if len(orig_shape) == 2:
                self._shape = orig_shape
            else:
                raise ValueError(f"Cannot infer matrix shape from {orig_shape}. Provide input_shape.")
        
        self._rows, self._cols = self._shape
        self._group_size = binary_tensor.config.group_size
        
        # Scales
        self._scales_fp32 = binary_tensor.scales.astype(np.float32)
        self._packed_data = binary_tensor.packed_data
        
        # Compute size info
        self._compressed_bytes = (binary_tensor.packed_data.nbytes + 
                                  binary_tensor.scales.nbytes)
        self._decompressed_bytes = self._rows * self._cols * 4  # FP32
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.uint8)
    
    @property
    def compressed_size_bytes(self) -> int:
        return self._compressed_bytes
    
    @property
    def decompressed_size_bytes(self) -> int:
        return self._decompressed_bytes
    
    def dequantize(self) -> np.ndarray:
        """Full dequantization to FP32 for reference."""
        return binary_dequantize(self._btensor).astype(np.float32)
    
    def matmul(self, x: np.ndarray, config: ComputationConfig) -> ComputeResult:
        """Direct binary GEMM on packed bits."""
        return BinaryMatmul().compute(self, x, config)


class BinaryMatmul(LowBitMatmul):
    """Binary direct matrix multiplication kernel using packed bits."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "binary_direct_packed"
    
    @property
    def supported_precisions(self) -> List[ComputePrecision]:
        return [ComputePrecision.FP32, ComputePrecision.FP16]
    
    def _unpack_bits(self, packed: np.ndarray, num_weights: int) -> np.ndarray:
        """Unpack bits to {-1, +1} int8 array."""
        unpacked = np.zeros(num_weights, dtype=np.int8)
        for i in range(num_weights):
            byte_idx = i // 8
            bit_idx = i % 8
            bit = (packed[byte_idx] >> bit_idx) & 1
            unpacked[i] = 1 if bit else -1
        return unpacked
    
    def compute(
        self,
        weight: BinaryMatrix,
        x: np.ndarray,
        config: ComputationConfig
    ) -> ComputeResult:
        """
        Compute Y = W @ X using packed binary representation.
        
        Binary weights: -1 (bit=0) or +1 (bit=1)
        Computation: Y = sum_g (scale_g * (sign_g @ X_g))
        
        Since binary weights are just {-1, +1}, the multiplication
        is effectively signed addition/subtraction.
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
            
            # Unpack weight slice for this group
            # Each row has ceil(K/8) bytes
            bytes_per_row = (K + 7) // 8
            group_bytes_per_row = (group_k + 7) // 8
            
            # Unpack group: (rows, group_k)
            w_group = np.zeros((weight._rows, group_k), dtype=np.int8)
            
            for r in range(weight._rows):
                row_start_byte = r * bytes_per_row + (k_start // 8)
                group_packed = packed[row_start_byte:row_start_byte + group_bytes_per_row]
                w_group[r, :] = self._unpack_bits(group_packed, group_k)
            
            # Extract input slice
            x_group = x[k_start:k_end, :]
            
            # Compute: {-1, +1} @ X = sum(where w=+1) - sum(where w=-1)
            # Can optimize: partial = pos_sum - neg_sum
            # But for correctness, just use matmul
            partial = w_group.astype(np.int32) @ x_group.astype(np.int32)
            
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
                temporary_bytes=weight._rows * group_k * 1,  # unpacked buffer
                peak_bytes=weight.compressed_size_bytes + x.nbytes + y.nbytes,
            ),
            timing_stats=TimingStats(
                preparation_time_s=prep_time,
                computation_time_s=compute_time,
                total_time_s=prep_time + compute_time,
            ),
            metadata={
                "method": "binary_direct_packed",
                "group_size": group_size,
                "num_groups": n_groups,
            }
        )


def binary_matmul_reference(
    weight: BinaryMatrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Reference: dequantize then matmul."""
    return create_reference_result(weight, x, config)


def binary_matmul_direct(
    weight: BinaryMatrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Direct binary matmul."""
    return BinaryMatmul().compute(weight, x, config)


def benchmark_binary(
    weight: BinaryMatrix,
    x: np.ndarray,
    config: ComputationConfig,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark binary direct vs reference."""
    # Warmup
    for _ in range(3):
        _ = binary_matmul_direct(weight, x, config)
        _ = binary_matmul_reference(weight, x, config)
    
    direct_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        direct = binary_matmul_direct(weight, x, config)
        direct_times.append(time.perf_counter() - start)
    
    ref_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ref = binary_matmul_reference(weight, x, config)
        ref_times.append(time.perf_counter() - start)
    
    direct = binary_matmul_direct(weight, x, config)
    ref = binary_matmul_reference(weight, x, config)
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