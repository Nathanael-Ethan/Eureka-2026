"""
LDMARK Ternary Direct Computation Kernel

Ternary weight matrix multiplication using packed 2-bit trits.
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
    TernaryTensor,
    TernaryConfig,
    TernaryEncoding,
    ternary_dequantize,
)


class TernaryMatrix(LowBitMatrix):
    """
    Ternary quantized matrix (weights in {-1, 0, +1}).
    
    Weights stored as packed 2-bit trits (00=-1, 01=0, 10=+1).
    """
    
    def __init__(
        self,
        ternary_tensor: TernaryTensor,
        input_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize from a TernaryTensor.
        
        Args:
            ternary_tensor: The ternary weight tensor
            input_shape: Optional explicit input shape (out_features, in_features)
        """
        self._ttensor = ternary_tensor
        
        # Determine matrix shape
        if input_shape is not None:
            self._shape = input_shape
        else:
            orig_shape = ternary_tensor.original_shape
            if len(orig_shape) == 2:
                self._shape = orig_shape
            else:
                raise ValueError(f"Cannot infer matrix shape from {orig_shape}. Provide input_shape.")
        
        self._rows, self._cols = self._shape
        self._group_size = ternary_tensor.config.group_size
        
        # Scales
        self._scales_fp32 = ternary_tensor.scales.astype(np.float32)
        self._packed_data = ternary_tensor.packed_data
        
        # Compute size info
        self._compressed_bytes = (ternary_tensor.packed_data.nbytes + 
                                  ternary_tensor.scales.nbytes)
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
        return ternary_dequantize(self._ttensor).astype(np.float32)
    
    def matmul(self, x: np.ndarray, config: ComputationConfig) -> ComputeResult:
        """Direct ternary GEMM on packed 2-bit representation."""
        return TernaryMatmul().compute(self, x, config)


class TernaryMatmul(LowBitMatmul):
    """Ternary direct matrix multiplication kernel using packed 2-bit trits."""
    
    def __init__(self):
        pass
    
    @property
    def name(self) -> str:
        return "ternary_direct_packed"
    
    @property
    def supported_precisions(self) -> List[ComputePrecision]:
        return [ComputePrecision.FP32, ComputePrecision.FP16]
    
    def _unpack_trits(self, packed: np.ndarray, num_weights: int) -> np.ndarray:
        """Unpack 2-bit trits to {-1, 0, +1} int8 array."""
        # 00 = -1, 01 = 0, 10 = +1
        unpacked = np.zeros(num_weights, dtype=np.int8)
        for i in range(num_weights):
            bit_pos = i * 2
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            
            if bit_offset <= 6:
                trit = (packed[byte_idx] >> bit_offset) & 0x3
            else:
                # Spans byte boundary
                trit = ((packed[byte_idx] >> bit_offset) | 
                       (packed[byte_idx + 1] << (8 - bit_offset))) & 0x3
            
            if trit == 0:
                unpacked[i] = -1
            elif trit == 1:
                unpacked[i] = 0
            else:  # trit == 2
                unpacked[i] = 1
        return unpacked
    
    def compute(
        self,
        weight: TernaryMatrix,
        x: np.ndarray,
        config: ComputationConfig
    ) -> ComputeResult:
        """
        Compute Y = W @ X using packed ternary representation.
        
        Ternary weights: -1 (00), 0 (01), +1 (10)
        Computation: Y = sum_g (scale_g * (trit_g @ X_g))
        
        Zero weights contribute nothing, so we only add/subtract.
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
            
            # Calculate packed data indices
            # 2 bits per weight = 4 weights per byte
            trits_per_byte = 4
            bytes_per_row = (K * 2 + 7) // 8
            group_bytes_per_row = (group_k * 2 + 7) // 8
            
            # Unpack group: (rows, group_k)
            w_group = np.zeros((weight._rows, group_k), dtype=np.int8)
            
            for r in range(weight._rows):
                row_start_byte = r * bytes_per_row + (k_start * 2 // 8)
                group_packed = packed[row_start_byte:row_start_byte + group_bytes_per_row]
                w_group[r, :] = self._unpack_trits(group_packed, group_k)
            
            # Extract input slice
            x_group = x[k_start:k_end, :]
            
            # Compute: {-1, 0, +1} @ X 
            # Zero weights contribute nothing
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
                temporary_bytes=weight._rows * group_k * 1,
                peak_bytes=weight.compressed_size_bytes + x.nbytes + y.nbytes,
            ),
            timing_stats=TimingStats(
                preparation_time_s=prep_time,
                computation_time_s=compute_time,
                total_time_s=prep_time + compute_time,
            ),
            metadata={
                "method": "ternary_direct_packed",
                "group_size": group_size,
                "num_groups": n_groups,
            }
        )


def ternary_matmul_reference(
    weight: TernaryMatrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Reference: dequantize then matmul."""
    return create_reference_result(weight, x, config)


def ternary_matmul_direct(
    weight: TernaryMatrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """Direct ternary matmul."""
    return TernaryMatmul().compute(weight, x, config)


def benchmark_ternary(
    weight: TernaryMatrix,
    x: np.ndarray,
    config: ComputationConfig,
    num_runs: int = 10
) -> Dict[str, Any]:
    """Benchmark ternary direct vs reference."""
    # Warmup
    for _ in range(3):
        _ = ternary_matmul_direct(weight, x, config)
        _ = ternary_matmul_reference(weight, x, config)
    
    direct_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        direct = ternary_matmul_direct(weight, x, config)
        direct_times.append(time.perf_counter() - start)
    
    ref_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ref = ternary_matmul_reference(weight, x, config)
        ref_times.append(time.perf_counter() - start)
    
    direct = ternary_matmul_direct(weight, x, config)
    ref = ternary_matmul_reference(weight, x, config)
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