"""
LDMARK Compiler - Transform Results

Data structures for compression transformation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ldmark.compression.quantize import QuantizedTensor, QuantizationConfig
from ldmark.compression.binary_ternary import BinaryTensor, TernaryTensor, BinaryConfig, TernaryConfig
from ldmark.compression.metrics import CompressionMetrics
from .plan import CompressionMethod


class CompressedTensorType(Enum):
    """Type of compressed tensor."""
    QUANTIZED = "quantized"
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass
class CompressedTensor:
    """Unified container for any compressed tensor."""
    tensor_name: str
    tensor_type: CompressedTensorType
    original_shape: tuple
    original_dtype: np.dtype
    original_num_elements: int

    # Quantized (INT8/INT4)
    quantized_data: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    quant_config: Optional[QuantizationConfig] = None

    # Binary
    binary_packed_data: Optional[np.ndarray] = None
    binary_scales: Optional[np.ndarray] = None
    binary_config: Optional[BinaryConfig] = None

    # Ternary
    ternary_packed_data: Optional[np.ndarray] = None
    ternary_scales: Optional[np.ndarray] = None
    ternary_config: Optional[TernaryConfig] = None

    # Metrics
    metrics: Optional[CompressionMetrics] = None

    # Validation
    dequantized: Optional[np.ndarray] = None
    validation_error: Optional[str] = None

    def get_compressed_size_bytes(self) -> int:
        """Get actual compressed storage size in bytes."""
        if self.tensor_type == CompressedTensorType.QUANTIZED:
            data_size = self.quantized_data.nbytes if self.quantized_data is not None else 0
            scales_size = self.scales.nbytes if self.scales is not None else 0
            return data_size + scales_size
        elif self.tensor_type == CompressedTensorType.BINARY:
            data_size = self.binary_packed_data.nbytes if self.binary_packed_data is not None else 0
            scales_size = self.binary_scales.nbytes if self.binary_scales is not None else 0
            return data_size + scales_size
        elif self.tensor_type == CompressedTensorType.TERNARY:
            data_size = self.ternary_packed_data.nbytes if self.ternary_packed_data is not None else 0
            scales_size = self.ternary_scales.nbytes if self.ternary_scales is not None else 0
            return data_size + scales_size
        return 0

    def get_original_size_bytes(self) -> int:
        """Get original tensor size in bytes."""
        return self.original_num_elements * self.original_dtype.itemsize

    def get_compression_ratio(self) -> float:
        """Get compression ratio (original / compressed)."""
        orig = self.get_original_size_bytes()
        comp = self.get_compressed_size_bytes()
        if comp == 0:
            return float('inf')
        return orig / comp

    def get_bits_per_weight(self) -> float:
        """Get effective bits per weight including overhead."""
        if self.metrics:
            return self.metrics.bits_per_weight
        comp_size = self.get_compressed_size_bytes()
        if comp_size == 0:
            return 0.0
        return (comp_size * 8) / self.original_num_elements

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "tensor_name": self.tensor_name,
            "tensor_type": self.tensor_type.value,
            "original_shape": self.original_shape,
            "original_dtype": str(self.original_dtype),
            "original_num_elements": self.original_num_elements,
            "original_size_bytes": self.get_original_size_bytes(),
            "compressed_size_bytes": self.get_compressed_size_bytes(),
            "compression_ratio": self.get_compression_ratio(),
            "bits_per_weight": self.get_bits_per_weight(),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "validation_error": self.validation_error,
        }
        if self.tensor_type == CompressedTensorType.QUANTIZED:
            base.update({
                "quantization_target": self.quant_config.target_bits.name if self.quant_config else None,
                "group_size": self.quant_config.group_size if self.quant_config else None,
                "scale_dtype": str(self.quant_config.scale_dtype) if self.quant_config else None,
                "symmetric": self.quant_config.symmetric if self.quant_config else None,
            })
        elif self.tensor_type == CompressedTensorType.BINARY:
            base.update({
                "encoding": self.binary_config.encoding.value if self.binary_config else None,
                "group_size": self.binary_config.group_size if self.binary_config else None,
                "scale_dtype": str(self.binary_config.scale_dtype) if self.binary_config else None,
            })
        elif self.tensor_type == CompressedTensorType.TERNARY:
            base.update({
                "encoding": self.ternary_config.encoding.value if self.ternary_config else None,
                "group_size": self.ternary_config.group_size if self.ternary_config else None,
                "scale_dtype": str(self.ternary_config.scale_dtype) if self.ternary_config else None,
            })
        return base


@dataclass
class TransformResult:
    """Result of the transform stage."""
    compressed_tensors: List[CompressedTensor] = field(default_factory=list)
    skipped_tensors: List[Dict[str, Any]] = field(default_factory=list)  # name, reason
    total_original_size_bytes: int = 0
    total_compressed_size_bytes: int = 0
    overall_compression_ratio: float = 0.0
    average_bits_per_weight: float = 0.0
    method: CompressionMethod = CompressionMethod.INT8
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compressed_tensors": [t.to_dict() for t in self.compressed_tensors],
            "skipped_tensors": self.skipped_tensors,
            "total_original_size_bytes": self.total_original_size_bytes,
            "total_compressed_size_bytes": self.total_compressed_size_bytes,
            "overall_compression_ratio": self.overall_compression_ratio,
            "average_bits_per_weight": self.average_bits_per_weight,
            "method": self.method.value,
            "warnings": self.warnings,
        }

    def get_tensor(self, name: str) -> Optional[CompressedTensor]:
        """Get compressed tensor by name."""
        for t in self.compressed_tensors:
            if t.tensor_name == name:
                return t
        return None