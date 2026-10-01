"""
LDMARK Compiler - Compression Planning

Data structures for compression strategy planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path

from ldmark.analysis.models import DType
from ldmark.compression.quantize import QuantizationTarget, QuantizationConfig
from ldmark.compression.binary_ternary import BinaryConfig, TernaryConfig


class CompressionMethod(Enum):
    """Available compression methods from Mission 03."""
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass(frozen=True)
class TensorCompressionPlan:
    """Compression plan for a single tensor."""
    tensor_name: str
    method: CompressionMethod
    original_shape: tuple
    original_dtype: DType
    original_num_elements: int
    original_size_bytes: int

    # Method-specific config
    target_bits: Optional[int] = None
    group_size: Optional[int] = None
    scale_dtype: Optional[str] = None
    symmetric: Optional[bool] = None
    encoding: Optional[str] = None
    threshold: Optional[float] = None

    # Expected results
    expected_bits_per_weight: float = 0.0
    expected_compressed_size_bytes: int = 0
    expected_compression_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "method": self.method.value,
            "original_shape": self.original_shape,
            "original_dtype": self.original_dtype.value,
            "original_num_elements": self.original_num_elements,
            "original_size_bytes": self.original_size_bytes,
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "symmetric": self.symmetric,
            "encoding": self.encoding,
            "threshold": self.threshold,
            "expected_bits_per_weight": self.expected_bits_per_weight,
            "expected_compressed_size_bytes": self.expected_compressed_size_bytes,
            "expected_compression_ratio": self.expected_compression_ratio,
        }


@dataclass
class CompressionPlan:
    """
    Complete compression plan for a model.

    Contains per-tensor plans and aggregate estimates.
    """
    method: CompressionMethod
    target_precision: str
    optimization_strategy: str
    target_hardware: str

    # Per-tensor plans
    tensor_plans: List[TensorCompressionPlan] = field(default_factory=list)

    # Aggregated estimates
    total_original_size_bytes: int = 0
    total_compressed_size_bytes: int = 0
    total_compression_ratio: float = 0.0
    average_bits_per_weight: float = 0.0

    # Configuration
    group_size: int = 128
    scale_dtype: str = "float16"
    symmetric: bool = True
    binary_encoding: str = "sign"
    ternary_threshold: float = 0.05

    # Metadata
    warnings: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "target_precision": self.target_precision,
            "optimization_strategy": self.optimization_strategy,
            "target_hardware": self.target_hardware,
            "tensor_plans": [p.to_dict() for p in self.tensor_plans],
            "total_original_size_bytes": self.total_original_size_bytes,
            "total_compressed_size_bytes": self.total_compressed_size_bytes,
            "total_compression_ratio": self.total_compression_ratio,
            "average_bits_per_weight": self.average_bits_per_weight,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "symmetric": self.symmetric,
            "binary_encoding": self.binary_encoding,
            "ternary_threshold": self.ternary_threshold,
            "warnings": self.warnings,
            "rationale": self.rationale,
        }

    def get_tensor_plan(self, tensor_name: str) -> Optional[TensorCompressionPlan]:
        """Get plan for a specific tensor."""
        for p in self.tensor_plans:
            if p.tensor_name == tensor_name:
                return p
        return None


def create_quantization_config(plan: TensorCompressionPlan) -> QuantizationConfig:
    """Create QuantizationConfig from tensor plan."""
    return QuantizationConfig(
        target_bits=QuantizationTarget(plan.target_bits) if plan.target_bits else QuantizationTarget.INT8,
        group_size=plan.group_size or 128,
        scale_dtype=plan.scale_dtype or np.float16,
        symmetric=plan.symmetric if plan.symmetric is not None else True,
    )


def create_binary_config(plan: TensorCompressionPlan) -> BinaryConfig:
    """Create BinaryConfig from tensor plan."""
    return BinaryConfig(
        encoding=plan.encoding or "sign",
        scale_dtype=plan.scale_dtype or np.float16,
        group_size=plan.group_size or 128,
    )


def create_ternary_config(plan: TensorCompressionPlan) -> TernaryConfig:
    """Create TernaryConfig from tensor plan."""
    return TernaryConfig(
        encoding=plan.encoding or "minus_zero_plus",
        scale_dtype=plan.scale_dtype or np.float16,
        group_size=plan.group_size or 128,
    )


import numpy as np