"""
Storage validation for LDMARK benchmark laboratory.

Verifies theoretical storage estimates against actual serialized storage,
accounting for metadata, scales, padding, tensor headers, and serialization overhead.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np

from .model import StorageMetrics


@dataclass(frozen=True)
class StorageValidationResult:
    """Result of storage validation comparing theoretical vs actual."""
    theoretical_bytes: int
    actual_bytes: int
    metadata_bytes: int
    scales_bytes: int
    padding_bytes: int
    tensor_headers_bytes: int
    serialization_overhead_bytes: int
    overhead_ratio: float  # actual / theoretical
    overhead_breakdown: Dict[str, float]  # Component -> bytes
    validation_passed: bool
    tolerance: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theoretical_bytes": self.theoretical_bytes,
            "actual_bytes": self.actual_bytes,
            "metadata_bytes": self.metadata_bytes,
            "scales_bytes": self.scales_bytes,
            "padding_bytes": self.padding_bytes,
            "tensor_headers_bytes": self.tensor_headers_bytes,
            "serialization_overhead_bytes": self.serialization_overhead_bytes,
            "overhead_ratio": self.overhead_ratio,
            "overhead_breakdown": self.overhead_breakdown,
            "validation_passed": self.validation_passed,
            "tolerance": self.tolerance,
        }


def calculate_theoretical_storage(
    num_weights: int,
    bits_per_weight: int,
    group_size: int,
    scale_bits: int = 16,
) -> int:
    """
    Calculate theoretical minimum storage in bytes.
    
    Formula: ceil((num_weights * bits_per_weight + num_groups * scale_bits) / 8)
    
    Args:
        num_weights: Total number of weights
        bits_per_weight: Bits per weight (1, 4, 8, 16, 32)
        group_size: Quantization group size
        scale_bits: Bits per scale factor (default 16 for FP16)
        
    Returns:
        Theoretical storage in bytes
    """
    if group_size <= 0:
        # No grouping (FP32, FP16)
        total_bits = num_weights * bits_per_weight
    else:
        num_groups = (num_weights + group_size - 1) // group_size
        total_bits = num_weights * bits_per_weight + num_groups * scale_bits
    
    return int(np.ceil(total_bits / 8))


def calculate_actual_storage(
    quantized_data: np.ndarray,
    scales: Optional[np.ndarray] = None,
    metadata_bytes: int = 0,
    tensor_header_bytes: int = 0,
    padding_bytes: int = 0,
) -> Dict[str, int]:
    """
    Calculate actual serialized storage from quantized components.
    
    Args:
        quantized_data: The quantized weight data array
        scales: Scale factors array (optional)
        metadata_bytes: Additional metadata bytes
        tensor_header_bytes: Per-tensor header bytes
        padding_bytes: Padding bytes for alignment
        
    Returns:
        Dictionary with breakdown of storage components
    """
    data_bytes = quantized_data.nbytes
    scales_bytes = scales.nbytes if scales is not None else 0
    
    return {
        "data_bytes": data_bytes,
        "scales_bytes": scales_bytes,
        "metadata_bytes": metadata_bytes,
        "tensor_header_bytes": tensor_header_bytes,
        "padding_bytes": padding_bytes,
        "total_bytes": data_bytes + scales_bytes + metadata_bytes + tensor_header_bytes + padding_bytes,
    }


def validate_storage(
    num_weights: int,
    bits_per_weight: int,
    group_size: int,
    quantized_data: np.ndarray,
    scales: Optional[np.ndarray] = None,
    scale_bits: int = 16,
    metadata_bytes: int = 0,
    tensor_header_bytes: int = 0,
    padding_bytes: int = 0,
    tolerance: float = 0.05,  # 5% tolerance
) -> StorageValidationResult:
    """
    Validate theoretical storage estimate against actual serialized storage.
    
    This is critical because bits/weight ≠ exact file size due to:
    - Metadata overhead
    - Scale factor storage
    - Padding for alignment
    - Tensor headers
    - Serialization format overhead
    
    Args:
        num_weights: Total number of weights
        bits_per_weight: Target bits per weight
        group_size: Quantization group size
        quantized_data: Actual quantized data array
        scales: Scale factors array
        scale_bits: Bits per scale factor
        metadata_bytes: Additional metadata bytes
        tensor_header_bytes: Per-tensor header bytes
        padding_bytes: Padding bytes
        tolerance: Acceptable overhead ratio tolerance (default 5%)
        
    Returns:
        StorageValidationResult with comparison details
    """
    theoretical = calculate_theoretical_storage(
        num_weights, bits_per_weight, group_size, scale_bits
    )
    
    actual_breakdown = calculate_actual_storage(
        quantized_data=quantized_data,
        scales=scales,
        metadata_bytes=metadata_bytes,
        tensor_header_bytes=tensor_header_bytes,
        padding_bytes=padding_bytes,
    )
    
    actual = actual_breakdown["total_bytes"]
    overhead_ratio = actual / theoretical if theoretical > 0 else 0.0
    
    # Check if actual is within tolerance of theoretical
    # Allow some overhead but flag if excessive
    validation_passed = overhead_ratio <= (1.0 + tolerance)
    
    overhead_breakdown = {
        "data": float(actual_breakdown["data_bytes"]),
        "scales": float(actual_breakdown["scales_bytes"]),
        "metadata": float(actual_breakdown["metadata_bytes"]),
        "tensor_headers": float(actual_breakdown["tensor_header_bytes"]),
        "padding": float(actual_breakdown["padding_bytes"]),
    }
    
    return StorageValidationResult(
        theoretical_bytes=theoretical,
        actual_bytes=actual,
        metadata_bytes=metadata_bytes,
        scales_bytes=actual_breakdown["scales_bytes"],
        padding_bytes=padding_bytes,
        tensor_headers_bytes=tensor_header_bytes,
        serialization_overhead_bytes=actual - theoretical,
        overhead_ratio=overhead_ratio,
        overhead_breakdown=overhead_breakdown,
        validation_passed=validation_passed,
        tolerance=tolerance,
    )


def validate_model_storage(
    tensor_results: List["TensorBenchmarkResult"],
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate storage for a complete model (multiple tensors).
    
    Args:
        tensor_results: List of tensor benchmark results
        tolerance: Overhead tolerance
        
    Returns:
        Aggregate validation results
    """
    from .model import TensorBenchmarkResult
    
    total_theoretical = sum(r.storage.theoretical_bytes for r in tensor_results)
    total_actual = sum(r.storage.actual_bytes for r in tensor_results)
    total_scales = sum(r.storage.scales_bytes for r in tensor_results)
    total_metadata = sum(r.storage.metadata_bytes for r in tensor_results)
    total_padding = sum(r.storage.padding_bytes for r in tensor_results)
    total_headers = sum(r.storage.tensor_headers_bytes for r in tensor_results)
    
    overhead_ratio = total_actual / total_theoretical if total_theoretical > 0 else 0.0
    validation_passed = overhead_ratio <= (1.0 + tolerance)
    
    per_tensor = {}
    for r in tensor_results:
        per_tensor[r.tensor_name] = {
            "theoretical_bytes": r.storage.theoretical_bytes,
            "actual_bytes": r.storage.actual_bytes,
            "overhead_ratio": r.storage.overhead_ratio,
            "scales_bytes": r.storage.scales_bytes,
            "validation_passed": r.storage.overhead_ratio <= (1.0 + tolerance),
        }
    
    return {
        "model_validation_passed": validation_passed,
        "total_theoretical_bytes": total_theoretical,
        "total_actual_bytes": total_actual,
        "total_overhead_ratio": overhead_ratio,
        "total_scales_bytes": total_scales,
        "total_metadata_bytes": total_metadata,
        "total_padding_bytes": total_padding,
        "total_tensor_headers_bytes": total_headers,
        "tolerance": tolerance,
        "per_tensor": per_tensor,
    }


def estimate_serialized_size(
    tensors: Dict[str, np.ndarray],
    quantized_tensors: Dict[str, Any],
    scales: Dict[str, np.ndarray],
    format_overhead_per_tensor: int = 64,  # bytes for safetensors-like header
) -> Dict[str, int]:
    """
    Estimate actual serialized file size for a model.
    
    Args:
        tensors: Original tensors
        quantized_tensors: Quantized tensor data (QuantizedTensor, BinaryTensor, etc.)
        scales: Scale factors per tensor
        format_overhead_per_tensor: Per-tensor serialization overhead
        
    Returns:
        Size breakdown in bytes
    """
    total_data = 0
    total_scales = 0
    total_headers = 0
    
    for name, qtensor in quantized_tensors.items():
        # Data bytes
        if hasattr(qtensor, 'data'):
            total_data += qtensor.data.nbytes
        elif hasattr(qtensor, 'packed_data'):
            total_data += qtensor.packed_data.nbytes
        
        # Scales bytes
        if name in scales:
            total_scales += scales[name].nbytes
        
        # Header overhead
        total_headers += format_overhead_per_tensor
    
    return {
        "quantized_data_bytes": total_data,
        "scales_bytes": total_scales,
        "format_headers_bytes": total_headers,
        "estimated_total_bytes": total_data + total_scales + total_headers,
    }