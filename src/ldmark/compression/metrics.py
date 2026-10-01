"""
Error metrics and compression ratio calculations for LDMARK compression laboratory.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class CompressionMetrics:
    """Container for compression experiment metrics."""
    original_dtype: str
    compressed_representation: str
    bits_per_weight: float
    scale_overhead_bits: float
    mean_absolute_error: float
    mean_squared_error: float
    max_absolute_error: float
    relative_error: float
    compression_ratio: float
    original_size_bytes: int
    compressed_size_bytes: int
    estimated_storage_bytes: int = 0
    actual_storage_bytes: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "original_dtype": self.original_dtype,
            "compressed_representation": self.compressed_representation,
            "bits_per_weight": self.bits_per_weight,
            "scale_overhead_bits": self.scale_overhead_bits,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "max_absolute_error": self.max_absolute_error,
            "relative_error": self.relative_error,
            "compression_ratio": self.compression_ratio,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "actual_storage_bytes": self.actual_storage_bytes,
        }


def calculate_error_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate error metrics between original and reconstructed tensors.
    
    Returns:
        (mae, mse, max_abs_error, relative_error)
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    
    mae = float(np.mean(abs_diff))
    mse = float(np.mean(diff ** 2))
    max_abs_error = float(np.max(abs_diff))
    
    # Relative error: ||x - x'|| / ||x||
    orig_norm = np.linalg.norm(original.astype(np.float64))
    if orig_norm > 0:
        relative_error = float(np.linalg.norm(diff) / orig_norm)
    else:
        relative_error = 0.0
    
    return mae, mse, max_abs_error, relative_error


def calculate_compression_ratio(
    original_size_bytes: int,
    compressed_size_bytes: int
) -> float:
    """Calculate compression ratio (original / compressed)."""
    if compressed_size_bytes == 0:
        return float('inf')
    return original_size_bytes / compressed_size_bytes


def calculate_storage_size(
    num_weights: int,
    bits_per_weight: float,
    num_scales: int = 0,
    scale_bits: int = 16
) -> int:
    """
    Calculate total storage size in bytes.
    
    Args:
        num_weights: Number of weights
        bits_per_weight: Bits per weight
        num_scales: Number of scale factors
        scale_bits: Bits per scale factor
        
    Returns:
        Size in bytes
    """
    weight_bits = num_weights * bits_per_weight
    scale_bits_total = num_scales * scale_bits
    total_bits = weight_bits + scale_bits_total
    return int(np.ceil(total_bits / 8))


def _to_dtype(scale_dtype) -> np.dtype:
    """Convert scale_dtype to numpy dtype."""
    return np.dtype(scale_dtype)


def compute_compression_metrics(
    original: np.ndarray,
    quantized_data: np.ndarray,
    scales: np.ndarray,
    target_bits: int,
    group_size: int,
    scale_dtype = np.float16
) -> CompressionMetrics:
    """
    Compute comprehensive compression metrics for an experiment.
    """
    scale_dtype = _to_dtype(scale_dtype)
    original_size = original.nbytes
    num_weights = original.size
    num_scales = len(scales)
    
    # Calculate bits per weight (including scale overhead)
    weight_bits = num_weights * target_bits
    scale_bits = num_scales * scale_dtype.itemsize * 8
    total_bits = weight_bits + scale_bits
    bits_per_weight = total_bits / num_weights
    
    # Compressed size in bytes (theoretical)
    compressed_size = int(np.ceil(total_bits / 8))
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    
    # Scale overhead
    scale_overhead_bits = scale_bits / num_weights
    
    # Error metrics - placeholders, caller should use compute_full_metrics
    mae = mse = max_err = rel_err = 0.0
    
    # Actual storage from quantized data + scales
    actual_storage = quantized_data.nbytes + scales.nbytes
    
    return CompressionMetrics(
        original_dtype=str(original.dtype),
        compressed_representation=f"INT{target_bits}_g{group_size}",
        bits_per_weight=bits_per_weight,
        scale_overhead_bits=scale_overhead_bits,
        mean_absolute_error=mae,
        mean_squared_error=mse,
        max_absolute_error=max_err,
        relative_error=rel_err,
        compression_ratio=compression_ratio,
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        estimated_storage_bytes=compressed_size,
        actual_storage_bytes=actual_storage,
    )


def compute_full_metrics(
    original: np.ndarray,
    dequantized: np.ndarray,
    quantized_data: np.ndarray,
    scales: np.ndarray,
    target_bits: int,
    group_size: int,
    scale_dtype = np.float16
) -> CompressionMetrics:
    """
    Compute full compression metrics including error metrics.
    """
    metrics = compute_compression_metrics(
        original, quantized_data, scales, target_bits, group_size, scale_dtype
    )
    
    mae, mse, max_err, rel_err = calculate_error_metrics(original, dequantized)
    
    return CompressionMetrics(
        original_dtype=metrics.original_dtype,
        compressed_representation=metrics.compressed_representation,
        bits_per_weight=metrics.bits_per_weight,
        scale_overhead_bits=metrics.scale_overhead_bits,
        mean_absolute_error=mae,
        mean_squared_error=mse,
        max_absolute_error=max_err,
        relative_error=rel_err,
        compression_ratio=metrics.compression_ratio,
        original_size_bytes=metrics.original_size_bytes,
        compressed_size_bytes=metrics.compressed_size_bytes,
        estimated_storage_bytes=metrics.estimated_storage_bytes,
        actual_storage_bytes=metrics.actual_storage_bytes,
    )