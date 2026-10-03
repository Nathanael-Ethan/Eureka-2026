"""
Mixed-precision plan data structures for LDMARK.

Defines the structured representation of a mixed-precision compilation plan.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import numpy as np
import uuid
from datetime import datetime


class PrecisionType(Enum):
    """Supported precision types in LDMARK."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"
    TERNARY = "ternary"
    
    @property
    def bits_per_weight(self) -> float:
        """Theoretical bits per weight (excluding overhead)."""
        mapping = {
            PrecisionType.FP32: 32.0,
            PrecisionType.FP16: 16.0,
            PrecisionType.INT8: 8.0,
            PrecisionType.INT4: 4.0,
            PrecisionType.BINARY: 1.0,
            PrecisionType.TERNARY: np.log2(3),  # ~1.585
        }
        return mapping[self]
    
    def is_quantized(self) -> bool:
        """Whether this is a quantized representation."""
        return self in (PrecisionType.INT8, PrecisionType.INT4, PrecisionType.BINARY, PrecisionType.TERNARY)


class TensorRole(Enum):
    """Known tensor roles in transformer architectures."""
    EMBEDDING = "embedding"
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_O = "attention_o"
    ATTENTION_QKV = "attention_qkv"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    MLP_GATE = "mlp_gate"
    NORM = "norm"
    OUTPUT_HEAD = "output_head"
    BIAS = "bias"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TensorClassification:
    """Classification of a tensor based on its properties."""
    tensor_name: str
    role: TensorRole
    shape: tuple
    num_parameters: int
    dtype: str
    layer_index: Optional[int] = None
    architecture: str = "unknown"
    is_weight: bool = True
    group_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "role": self.role.value,
            "shape": self.shape,
            "num_parameters": self.num_parameters,
            "dtype": self.dtype,
            "layer_index": self.layer_index,
            "architecture": self.architecture,
            "is_weight": self.is_weight,
            "group_size": self.group_size,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PrecisionCandidate:
    """A candidate precision representation for a tensor."""
    precision: PrecisionType
    group_size: Optional[int] = None
    codebook_size: Optional[int] = None  # For codebook-based quantization
    estimated_bits_per_weight: float = 0.0
    estimated_storage_bytes: int = 0
    
    def __post_init__(self):
        if self.estimated_bits_per_weight == 0.0:
            # Calculate from precision type and group size
            if self.precision.is_quantized() and self.group_size:
                # Group-wise quantization: bits + scale overhead
                scale_bits = 16  # FP16 scale
                self.estimated_bits_per_weight = self.precision.bits_per_weight + (scale_bits / self.group_size)
            else:
                self.estimated_bits_per_weight = self.precision.bits_per_weight
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision.value,
            "group_size": self.group_size,
            "codebook_size": self.codebook_size,
            "estimated_bits_per_weight": self.estimated_bits_per_weight,
            "estimated_storage_bytes": self.estimated_storage_bytes,
        }


@dataclass(frozen=True)
class TensorPrecisionAssignment:
    """Assignment of a precision to a specific tensor."""
    tensor_name: str
    precision: PrecisionType
    group_size: Optional[int] = None
    codebook_size: Optional[int] = None
    candidate: Optional[PrecisionCandidate] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "precision": self.precision.value,
            "group_size": self.group_size,
            "codebook_size": self.codebook_size,
            "candidate": self.candidate.to_dict() if self.candidate else None,
        }
    
    @property
    def key(self) -> str:
        return self.tensor_name


@dataclass(frozen=True)
class MixedPrecisionPlan:
    """Complete mixed-precision compilation plan for a model."""
    model_identifier: str
    assignments: Dict[str, TensorPrecisionAssignment] = field(default_factory=dict)
    global_constraints: Dict[str, Any] = field(default_factory=dict)
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    
    def add_assignment(self, assignment: TensorPrecisionAssignment) -> "MixedPrecisionPlan":
        """Add an assignment (returns new plan since immutable)."""
        new_assignments = dict(self.assignments)
        new_assignments[assignment.tensor_name] = assignment
        return MixedPrecisionPlan(
            model_identifier=self.model_identifier,
            assignments=new_assignments,
            global_constraints=self.global_constraints,
            experiment_id=self.experiment_id,
            timestamp=self.timestamp,
            notes=self.notes,
        )
    
    def get_assignment(self, tensor_name: str) -> Optional[TensorPrecisionAssignment]:
        return self.assignments.get(tensor_name)
    
    def get_precision(self, tensor_name: str) -> Optional[PrecisionType]:
        assignment = self.assignments.get(tensor_name)
        return assignment.precision if assignment else None
    
    def tensors_by_precision(self, precision: PrecisionType) -> List[str]:
        return [name for name, a in self.assignments.items() if a.precision == precision]
    
    def total_tensors(self) -> int:
        return len(self.assignments)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "assignments": {k: v.to_dict() for k, v in self.assignments.items()},
            "global_constraints": self.global_constraints,
            "notes": self.notes,
            "total_tensors": self.total_tensors(),
            "precision_distribution": {
                p.value: len(self.tensors_by_precision(p)) for p in PrecisionType
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


def create_mixed_precision_plan(
    model_identifier: str,
    assignments: Dict[str, TensorPrecisionAssignment],
    global_constraints: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
) -> MixedPrecisionPlan:
    """Factory function to create a mixed-precision plan."""
    return MixedPrecisionPlan(
        model_identifier=model_identifier,
        assignments=assignments,
        global_constraints=global_constraints or {},
        experiment_id=experiment_id or str(uuid.uuid4())[:8],
    )


def create_uniform_plan(
    model_identifier: str,
    tensor_names: List[str],
    precision: PrecisionType,
    group_size: Optional[int] = None,
) -> MixedPrecisionPlan:
    """Create a uniform precision plan (all tensors same precision)."""
    assignments = {}
    for name in tensor_names:
        candidate = PrecisionCandidate(
            precision=precision,
            group_size=group_size,
        )
        assignments[name] = TensorPrecisionAssignment(
            tensor_name=name,
            precision=precision,
            group_size=group_size,
            candidate=candidate,
        )
    return create_mixed_precision_plan(model_identifier, assignments)


def get_precision_candidates(
    tensor_classification: TensorClassification,
    allowed_precisions: Optional[List[PrecisionType]] = None,
    default_group_size: int = 128,
) -> List[PrecisionCandidate]:
    """Get all valid precision candidates for a tensor."""
    if allowed_precisions is None:
        allowed_precisions = [
            PrecisionType.FP16,
            PrecisionType.INT8,
            PrecisionType.INT4,
            PrecisionType.BINARY,
            PrecisionType.TERNARY,
        ]
    
    candidates = []
    num_params = tensor_classification.num_parameters
    
    for prec in allowed_precisions:
        if prec == PrecisionType.FP16:
            # No grouping for FP16
            candidates.append(PrecisionCandidate(
                precision=prec,
                estimated_bits_per_weight=prec.bits_per_weight,
                estimated_storage_bytes=num_params * 2,
            ))
        elif prec == PrecisionType.INT8:
            # Group-wise INT8
            candidates.append(PrecisionCandidate(
                precision=prec,
                group_size=default_group_size,
                estimated_bits_per_weight=prec.bits_per_weight + (16 / default_group_size),
                estimated_storage_bytes=int(num_params * (prec.bits_per_weight + 16 / default_group_size) / 8),
            ))
        elif prec == PrecisionType.INT4:
            # Group-wise INT4
            candidates.append(PrecisionCandidate(
                precision=prec,
                group_size=default_group_size,
                estimated_bits_per_weight=prec.bits_per_weight + (16 / default_group_size),
                estimated_storage_bytes=int(num_params * (prec.bits_per_weight + 16 / default_group_size) / 8),
            ))
        elif prec == PrecisionType.BINARY:
            # Binary: 1 bit + scale overhead
            candidates.append(PrecisionCandidate(
                precision=prec,
                group_size=default_group_size,
                estimated_bits_per_weight=1.0 + (16 / default_group_size),
                estimated_storage_bytes=int(num_params * (1.0 + 16 / default_group_size) / 8),
            ))
        elif prec == PrecisionType.TERNARY:
            # Ternary: ~1.585 bits + scale overhead
            candidates.append(PrecisionCandidate(
                precision=prec,
                group_size=default_group_size,
                estimated_bits_per_weight=np.log2(3) + (16 / default_group_size),
                estimated_storage_bytes=int(num_params * (np.log2(3) + 16 / default_group_size) / 8),
            ))
    
    return candidates