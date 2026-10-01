from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from .formats import ModelFormat


class TensorDtype(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT4 = "int4"
    UINT8 = "uint8"
    FLOAT8 = "float8"
    FLOAT4 = "float4"
    BOOL = "bool"
    UNKNOWN = "unknown"


@dataclass
class TensorMetadata:
    name: str
    shape: List[int]
    dtype: TensorDtype
    num_elements: int
    raw_bytes: int
    offset: Optional[int] = None
    length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype.value,
            "num_elements": self.num_elements,
            "raw_bytes": self.raw_bytes,
        }


@dataclass
class ModelMetadata:
    model_id: str
    source_path: str
    source_format: ModelFormat
    architecture: str = "unknown"
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context_length: Optional[int] = None
    attention_type: Optional[str] = None
    total_parameters: int = 0
    tensor_count: int = 0
    tensors: List[TensorMetadata] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_path": self.source_path,
            "source_format": self.source_format.value,
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "max_context_length": self.max_context_length,
            "attention_type": self.attention_type,
            "total_parameters": self.total_parameters,
            "tensor_count": self.tensor_count,
            "tensors": [t.to_dict() for t in self.tensors],
            "config": self.config,
            "warnings": self.warnings,
        }

    def get_architecture_summary(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "max_context_length": self.max_context_length,
            "attention_type": self.attention_type,
        }