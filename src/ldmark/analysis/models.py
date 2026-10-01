from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class DType(Enum):
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "float8"
    FP4 = "float4"
    BIT1 = "1bit"

    @property
    def bits(self) -> int:
        return {
            DType.FP32: 32,
            DType.FP16: 16,
            DType.BF16: 16,
            DType.INT8: 8,
            DType.INT4: 4,
            DType.FP8: 8,
            DType.FP4: 4,
            DType.BIT1: 1,
        }[self]

    @property
    def bytes_per_element(self) -> float:
        return self.bits / 8.0


class TensorTrainableStatus(Enum):
    TRAINABLE = "trainable"
    FROZEN = "frozen"
    UNKNOWN = "unknown"


@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: DType
    num_elements: int
    estimated_raw_storage_bytes: int
    trainable: TensorTrainableStatus = TensorTrainableStatus.UNKNOWN

    @classmethod
    def from_shape_and_dtype(cls, name: str, shape: List[int], dtype: DType, trainable: TensorTrainableStatus = TensorTrainableStatus.UNKNOWN) -> TensorInfo:
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        estimated_raw_storage_bytes = int(num_elements * dtype.bytes_per_element)
        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            num_elements=num_elements,
            estimated_raw_storage_bytes=estimated_raw_storage_bytes,
            trainable=trainable,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype.value,
            "num_elements": self.num_elements,
            "estimated_raw_storage_bytes": self.estimated_raw_storage_bytes,
            "trainable": self.trainable.value,
        }


@dataclass
class ParameterCounts:
    total: int = 0
    by_dtype: Dict[DType, int] = field(default_factory=dict)
    by_tensor: Dict[str, int] = field(default_factory=dict)
    trainable: int = 0
    non_trainable: int = 0
    unknown_trainable: int = 0

    def add_tensor(self, name: str, count: int, dtype: DType, trainable: TensorTrainableStatus):
        self.total += count
        self.by_dtype[dtype] = self.by_dtype.get(dtype, 0) + count
        self.by_tensor[name] = count
        if trainable == TensorTrainableStatus.TRAINABLE:
            self.trainable += count
        elif trainable == TensorTrainableStatus.FROZEN:
            self.non_trainable += count
        else:
            self.unknown_trainable += count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "by_dtype": {k.value: v for k, v in self.by_dtype.items()},
            "by_tensor": self.by_tensor,
            "trainable": self.trainable,
            "non_trainable": self.non_trainable,
            "unknown_trainable": self.unknown_trainable,
        }


@dataclass
class ArchitectureInfo:
    model_architecture: Optional[str] = None
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context_length: Optional[int] = None
    attention_type: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if key == "additional_metadata":
                    result.update(value)
                else:
                    result[key] = value
        return result


@dataclass
class StorageEstimate:
    dtype: DType
    estimated_bytes: int
    estimated_gb: float
    is_theoretical: bool = True
    description: str = ""

    @classmethod
    def from_parameter_count(cls, param_count: int, dtype: DType, description: str = "") -> StorageEstimate:
        estimated_bytes = int(param_count * dtype.bytes_per_element)
        estimated_gb = estimated_bytes / (1024 ** 3)
        return cls(
            dtype=dtype,
            estimated_bytes=estimated_bytes,
            estimated_gb=estimated_gb,
            is_theoretical=True,
            description=description or f"Theoretical {dtype.value} weight storage",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype.value,
            "estimated_bytes": self.estimated_bytes,
            "estimated_gb": round(self.estimated_gb, 4),
            "is_theoretical": self.is_theoretical,
            "description": self.description,
        }


@dataclass
class CompressionEstimate:
    format_name: str
    effective_bits_per_weight: float
    estimated_bytes: int
    estimated_gb: float
    compression_ratio: float
    group_size: Optional[int] = None
    scale_dtype: Optional[DType] = None
    description: str = ""

    @classmethod
    def from_parameters(
        cls,
        param_count: int,
        format_name: str,
        effective_bits_per_weight: float,
        baseline_dtype: DType = DType.FP16,
        group_size: Optional[int] = None,
        scale_dtype: Optional[DType] = None,
        description: str = "",
    ) -> CompressionEstimate:
        baseline_bytes = int(param_count * baseline_dtype.bytes_per_element)
        estimated_bytes = int(param_count * effective_bits_per_weight / 8.0)
        if group_size and scale_dtype:
            num_groups = (param_count + group_size - 1) // group_size
            scale_bytes = num_groups * scale_dtype.bytes_per_element
            estimated_bytes += scale_bytes
        estimated_gb = estimated_bytes / (1024 ** 3)
        compression_ratio = baseline_bytes / estimated_bytes if estimated_bytes > 0 else 0.0
        return cls(
            format_name=format_name,
            effective_bits_per_weight=effective_bits_per_weight,
            estimated_bytes=estimated_bytes,
            estimated_gb=estimated_gb,
            compression_ratio=compression_ratio,
            group_size=group_size,
            scale_dtype=scale_dtype,
            description=description,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_name": self.format_name,
            "effective_bits_per_weight": self.effective_bits_per_weight,
            "estimated_bytes": self.estimated_bytes,
            "estimated_gb": round(self.estimated_gb, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype.value if self.scale_dtype else None,
            "description": self.description,
        }


@dataclass
class RuntimeMemoryEstimate:
    weights_gb: float
    kv_cache_gb: float
    activations_gb: float
    overhead_gb: float
    total_gb: float
    assumptions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def estimate(
        cls,
        param_count: int,
        weight_dtype: DType,
        context_length: int,
        hidden_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        batch_size: int = 1,
        activation_dtype: DType = DType.FP16,
    ) -> RuntimeMemoryEstimate:
        weights_bytes = param_count * weight_dtype.bytes_per_element
        weights_gb = weights_bytes / (1024 ** 3)

        kv_cache_elements = 2 * batch_size * num_layers * context_length * num_kv_heads * head_dim
        kv_cache_bytes = kv_cache_elements * activation_dtype.bytes_per_element
        kv_cache_gb = kv_cache_bytes / (1024 ** 3)

        activations_per_layer = batch_size * context_length * hidden_size * 4
        total_activation_elements = activations_per_layer * num_layers
        activations_bytes = total_activation_elements * activation_dtype.bytes_per_element
        activations_gb = activations_bytes / (1024 ** 3)

        overhead_gb = (weights_gb + kv_cache_gb + activations_gb) * 0.1
        total_gb = weights_gb + kv_cache_gb + activations_gb + overhead_gb

        return cls(
            weights_gb=round(weights_gb, 4),
            kv_cache_gb=round(kv_cache_gb, 4),
            activations_gb=round(activations_gb, 4),
            overhead_gb=round(overhead_gb, 4),
            total_gb=round(total_gb, 4),
            assumptions={
                "batch_size": batch_size,
                "context_length": context_length,
                "activation_dtype": activation_dtype.value,
                "note": "Estimates are approximate and architecture-dependent",
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights_gb": self.weights_gb,
            "kv_cache_gb": self.kv_cache_gb,
            "activations_gb": self.activations_gb,
            "overhead_gb": self.overhead_gb,
            "total_gb": self.total_gb,
            "assumptions": self.assumptions,
        }


@dataclass
class ModelAnalysis:
    model_id: str
    source_path: str
    parameter_counts: ParameterCounts
    tensors: List[TensorInfo]
    architecture: ArchitectureInfo
    storage_estimates: List[StorageEstimate]
    compression_estimates: List[CompressionEstimate]
    runtime_memory: Optional[RuntimeMemoryEstimate] = None
    actual_file_size_bytes: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_path": self.source_path,
            "parameter_counts": self.parameter_counts.to_dict(),
            "tensors": [t.to_dict() for t in self.tensors],
            "architecture": self.architecture.to_dict(),
            "storage_estimates": [s.to_dict() for s in self.storage_estimates],
            "compression_estimates": [c.to_dict() for c in self.compression_estimates],
            "runtime_memory": self.runtime_memory.to_dict() if self.runtime_memory else None,
            "actual_file_size_bytes": self.actual_file_size_bytes,
            "warnings": self.warnings,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)