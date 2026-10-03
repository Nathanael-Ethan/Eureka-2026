"""
LDMARK Low-Bit Computation Engine - Core Abstractions

Base interfaces for direct computation on compressed representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import time


class ComputePrecision(Enum):
    """Precision of the computation result."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class ComputationConfig:
    """Configuration for low-bit computation."""
    output_precision: ComputePrecision = ComputePrecision.FP32
    accumulate_in_fp32: bool = True  # Use FP32 accumulation for numerical stability
    batch_size: int = 1
    enable_memory_tracking: bool = True


@dataclass
class MemoryStats:
    """Memory usage statistics for a computation."""
    compressed_weight_bytes: int = 0
    decompressed_weight_bytes: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    temporary_bytes: int = 0
    peak_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compressed_weight_bytes": self.compressed_weight_bytes,
            "decompressed_weight_bytes": self.decompressed_weight_bytes,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "temporary_bytes": self.temporary_bytes,
            "peak_bytes": self.peak_bytes,
        }


@dataclass
class TimingStats:
    """Timing statistics for a computation."""
    preparation_time_s: float = 0.0
    computation_time_s: float = 0.0
    total_time_s: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preparation_time_s": self.preparation_time_s,
            "computation_time_s": self.computation_time_s,
            "total_time_s": self.total_time_s,
        }


@dataclass
class ComputeResult:
    """Result of a low-bit matrix multiplication."""
    output: np.ndarray
    memory_stats: MemoryStats
    timing_stats: TimingStats
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LowBitMatrix(ABC):
    """
    Abstract base class for low-bit weight matrices.
    
    Represents a compressed weight matrix that can participate in
    matrix multiplication without full dequantization.
    """
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Shape of the matrix (rows, cols)."""
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Storage dtype of the compressed data."""
        pass
    
    @property
    @abstractmethod
    def compressed_size_bytes(self) -> int:
        """Size of compressed representation in bytes."""
        pass
    
    @property
    @abstractmethod
    def decompressed_size_bytes(self) -> int:
        """Size if fully decompressed to FP32."""
        pass
    
    @abstractmethod
    def dequantize(self) -> np.ndarray:
        """Fully dequantize to FP32 (for reference comparison)."""
        pass
    
    @abstractmethod
    def matmul(self, x: np.ndarray, config: ComputationConfig) -> ComputeResult:
        """
        Compute Y = W @ X directly on compressed representation.
        
        Args:
            x: Input matrix of shape (K, N) or vector of shape (K,)
            config: Computation configuration
            
        Returns:
            ComputeResult with output and statistics
        """
        pass


class LowBitMatmul(ABC):
    """
    Abstract interface for low-bit matrix multiplication kernels.
    
    Separates the computation logic from the weight representation,
    enabling different backends (NumPy, CUDA, etc.).
    """
    
    @abstractmethod
    def compute(
        self,
        weight: LowBitMatrix,
        x: np.ndarray,
        config: ComputationConfig
    ) -> ComputeResult:
        """
        Compute Y = W @ X.
        
        Args:
            weight: Low-bit weight matrix
            x: Input matrix/vector
            config: Computation configuration
            
        Returns:
            ComputeResult
        """
        pass
    
    @property
    @abstractmethod
    def supported_precisions(self) -> List[ComputePrecision]:
        """List of supported output precisions."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this kernel."""
        pass


def create_reference_result(
    weight: LowBitMatrix,
    x: np.ndarray,
    config: ComputationConfig
) -> ComputeResult:
    """
    Reference implementation: dequantize then matmul.
    
    This provides the correctness baseline.
    """
    start_prep = time.perf_counter()
    w_fp32 = weight.dequantize()
    prep_time = time.perf_counter() - start_prep
    
    start_compute = time.perf_counter()
    output = w_fp32 @ x
    compute_time = time.perf_counter() - start_compute
    
    if config.output_precision == ComputePrecision.FP16:
        output = output.astype(np.float16)
    
    return ComputeResult(
        output=output,
        memory_stats=MemoryStats(
            compressed_weight_bytes=weight.compressed_size_bytes,
            decompressed_weight_bytes=weight.decompressed_size_bytes,
            input_bytes=x.nbytes,
            output_bytes=output.nbytes,
            temporary_bytes=w_fp32.nbytes,
            peak_bytes=w_fp32.nbytes + x.nbytes + output.nbytes,
        ),
        timing_stats=TimingStats(
            preparation_time_s=prep_time,
            computation_time_s=compute_time,
            total_time_s=prep_time + compute_time,
        ),
        metadata={"method": "reference_dequantize_then_matmul"}
    )


def compare_results(
    direct: ComputeResult,
    reference: ComputeResult,
    rtol: float = 1e-5,
    atol: float = 1e-7
) -> Dict[str, Any]:
    """
    Compare direct computation result against reference.
    
    Returns dictionary with error metrics.
    """
    direct_out = direct.output.astype(np.float32)
    ref_out = reference.output.astype(np.float32)
    
    abs_error = np.abs(direct_out - ref_out)
    max_abs_error = np.max(abs_error)
    mean_abs_error = np.mean(abs_error)
    
    # Relative error (avoid division by zero)
    rel_error = abs_error / (np.abs(ref_out) + 1e-12)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)
    
    # Check if within tolerance
    passed = np.allclose(direct_out, ref_out, rtol=rtol, atol=atol)
    
    return {
        "passed": bool(passed),
        "max_absolute_error": float(max_abs_error),
        "mean_absolute_error": float(mean_abs_error),
        "max_relative_error": float(max_rel_error),
        "mean_relative_error": float(mean_rel_error),
        "rtol": rtol,
        "atol": atol,
        "output_shape": direct_out.shape,
        "reference_shape": ref_out.shape,
    }


def track_memory(fn):
    """Decorator to track memory usage of a function."""
    import tracemalloc
    
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = fn(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        if hasattr(result, 'memory_stats'):
            result.memory_stats.temporary_bytes = current
            result.memory_stats.peak_bytes = peak
        
        return result
    return wrapper