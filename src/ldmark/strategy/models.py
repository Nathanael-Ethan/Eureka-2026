"""
LDMARK Strategy Engine - Core Models

Structured objects representing proposed compilation strategies.
This is the DECISION LAYER - it determines WHAT SHOULD BE DONE, not HOW TO DO IT.

Does NOT perform compression. Does NOT rank strategies. Does NOT claim quality retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class Backend(Enum):
    """Hardware compute backends."""
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> Backend:
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class WeightRepresentation(Enum):
    """Weight representation formats."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BINARY = "binary"
    TERNARY = "ternary"
    LDMARK_BINARY = "ldmark_binary"
    LDMARK_TERNARY = "ldmark_ternary"
    LDMARK_INT4 = "ldmark_int4"
    LDMARK_INT8 = "ldmark_int8"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, value: str) -> WeightRepresentation:
        try:
            return cls(value.lower())
        except ValueError:
            return cls.CUSTOM


class FeasibilityState(Enum):
    """Feasibility state for a strategy component."""
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StorageFeasibility:
    """Feasibility of model storage requirements."""
    state: FeasibilityState
    estimated_storage_bytes: int
    estimated_storage_gb: float
    constraint_limit_bytes: Optional[int]
    explanation: str
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "estimated_storage_gb": round(self.estimated_storage_gb, 4),
            "constraint_limit_bytes": self.constraint_limit_bytes,
            "explanation": self.explanation,
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class RuntimeMemoryFeasibility:
    """Feasibility of runtime memory requirements."""
    state: FeasibilityState
    estimated_weights_bytes: int
    estimated_kv_cache_bytes: int
    estimated_activations_bytes: int
    estimated_overhead_bytes: int
    estimated_total_bytes: int
    estimated_total_gb: float
    constraint_limit_bytes: Optional[int]
    available_memory_bytes: Optional[int]
    memory_pressure_ratio: float
    explanation: str
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "estimated_weights_bytes": self.estimated_weights_bytes,
            "estimated_kv_cache_bytes": self.estimated_kv_cache_bytes,
            "estimated_activations_bytes": self.estimated_activations_bytes,
            "estimated_overhead_bytes": self.estimated_overhead_bytes,
            "estimated_total_bytes": self.estimated_total_bytes,
            "estimated_total_gb": round(self.estimated_total_gb, 4),
            "constraint_limit_bytes": self.constraint_limit_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_pressure_ratio": round(self.memory_pressure_ratio, 4),
            "explanation": self.explanation,
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class BackendCompatibility:
    """Backend compatibility for a strategy."""
    backend: Backend
    state: FeasibilityState
    has_kernel: bool
    explanation: str
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "state": self.state.value,
            "has_kernel": self.has_kernel,
            "explanation": self.explanation,
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class CompilationStrategy:
    """
    A proposed compilation strategy with full feasibility analysis.

    Contains factual calculations only. No quality claims. No rankings.
    """
    strategy_id: str
    weight_representation: WeightRepresentation
    bits_per_weight: float
    group_size: Optional[int]
    scale_dtype: Optional[str]
    scale_bits: int

    storage_feasibility: StorageFeasibility
    runtime_feasibility: RuntimeMemoryFeasibility
    backend_compatibility: List[BackendCompatibility]

    warnings: List[str] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "weight_representation": self.weight_representation.value,
            "bits_per_weight": round(self.bits_per_weight, 4),
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "scale_bits": self.scale_bits,
            "storage_feasibility": self.storage_feasibility.to_dict(),
            "runtime_feasibility": self.runtime_feasibility.to_dict(),
            "backend_compatibility": [b.to_dict() for b in self.backend_compatibility],
            "warnings": self.warnings,
            "assumptions": self.assumptions,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def is_storage_feasible(self) -> bool:
        return self.storage_feasibility.state == FeasibilityState.FEASIBLE

    @property
    def is_runtime_feasible(self) -> bool:
        return self.runtime_feasibility.state == FeasibilityState.FEASIBLE

    @property
    def is_fully_feasible(self) -> bool:
        return self.is_storage_feasible and self.is_runtime_feasible

    @property
    def has_runtime_support(self) -> bool:
        return any(b.state == FeasibilityState.FEASIBLE and b.has_kernel for b in self.backend_compatibility)


@dataclass(frozen=True)
class StrategySet:
    """Complete set of evaluated strategies for a model/hardware/constraint combination."""
    model_id: str
    parameter_count: int
    baseline_dtype: str
    target_hardware: str
    strategies: List[CompilationStrategy]
    constraints_used: Dict[str, Any]
    context_length: int
    batch_size: int
    timestamp: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_count": self.parameter_count,
            "baseline_dtype": self.baseline_dtype,
            "target_hardware": self.target_hardware,
            "strategies": [s.to_dict() for s in self.strategies],
            "constraints_used": self.constraints_used,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def feasible_strategies(self) -> List[CompilationStrategy]:
        return [s for s in self.strategies if s.is_fully_feasible]

    @property
    def infeasible_strategies(self) -> List[CompilationStrategy]:
        return [s for s in self.strategies if not s.is_fully_feasible]

    @property
    def storage_feasible_only(self) -> List[CompilationStrategy]:
        return [s for s in self.strategies if s.is_storage_feasible and not s.is_runtime_feasible]


@dataclass(frozen=True)
class ConstraintConfig:
    """User constraints for strategy evaluation."""
    max_model_storage_bytes: Optional[int] = None
    max_runtime_memory_bytes: Optional[int] = None
    minimum_available_memory_bytes: Optional[int] = None
    preferred_precision: Optional[str] = None
    preferred_backend: Optional[str] = None
    max_context_length: Optional[int] = None
    batch_size: int = 1
    target_backends: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_storage_bytes": self.max_model_storage_bytes,
            "max_runtime_memory_bytes": self.max_runtime_memory_bytes,
            "minimum_available_memory_bytes": self.minimum_available_memory_bytes,
            "preferred_precision": self.preferred_precision,
            "preferred_backend": self.preferred_backend,
            "max_context_length": self.max_context_length,
            "batch_size": self.batch_size,
            "target_backends": self.target_backends,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConstraintConfig:
        return cls(
            max_model_storage_bytes=data.get("max_model_storage_bytes"),
            max_runtime_memory_bytes=data.get("max_runtime_memory_bytes"),
            minimum_available_memory_bytes=data.get("minimum_available_memory_bytes"),
            preferred_precision=data.get("preferred_precision"),
            preferred_backend=data.get("preferred_backend"),
            max_context_length=data.get("max_context_length"),
            batch_size=data.get("batch_size", 1),
            target_backends=data.get("target_backends", []),
        )


@dataclass(frozen=True)
class HardwareContext:
    """Hardware context for strategy evaluation."""
    total_ram_bytes: int
    available_ram_bytes: Optional[int]
    gpu_vram_bytes: Optional[int]
    supported_backends: List[Backend]
    cpu_architecture: str
    gpu_model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "gpu_vram_bytes": self.gpu_vram_bytes,
            "supported_backends": [b.value for b in self.supported_backends],
            "cpu_architecture": self.cpu_architecture,
            "gpu_model": self.gpu_model,
        }

    @classmethod
    def from_hardware_profile(cls, profile: "HardwareProfile") -> HardwareContext:
        from src.ldmark.hardware.profile import HardwareProfile, GPURuntime
        
        backends = []
        if profile.runtime.supported_backends:
            backends = [Backend.from_string(b.value) for b in profile.runtime.supported_backends]
        elif profile.gpu and profile.gpu.runtime != GPURuntime.NONE:
            backends = [Backend.from_string(profile.gpu.runtime.value)]
        else:
            backends = [Backend.CPU]
        
        return cls(
            total_ram_bytes=profile.system.total_ram_bytes,
            available_ram_bytes=profile.system.available_ram_bytes,
            gpu_vram_bytes=profile.gpu.vram_bytes if profile.gpu else None,
            supported_backends=backends,
            cpu_architecture=profile.cpu.architecture.value if profile.cpu.architecture else "unknown",
            gpu_model=profile.gpu.model if profile.gpu else None,
        )


@dataclass(frozen=True)
class ModelContext:
    """Model context for strategy evaluation."""
    model_id: str
    parameter_count: int
    architecture_name: Optional[str]
    num_layers: Optional[int]
    hidden_size: Optional[int]
    num_attention_heads: Optional[int]
    num_kv_heads: Optional[int]
    intermediate_size: Optional[int]
    vocab_size: Optional[int]
    max_context_length: Optional[int]
    attention_type: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_count": self.parameter_count,
            "architecture_name": self.architecture_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_context_length": self.max_context_length,
            "attention_type": self.attention_type,
        }

    @classmethod
    def from_model_analysis(cls, analysis: "ModelAnalysis") -> ModelContext:
        from src.ldmark.analysis.models import ModelAnalysis
        
        arch = analysis.architecture
        return cls(
            model_id=analysis.model_id,
            parameter_count=analysis.parameter_counts.total,
            architecture_name=arch.model_architecture,
            num_layers=arch.num_layers,
            hidden_size=arch.hidden_size,
            num_attention_heads=arch.num_attention_heads,
            num_kv_heads=arch.num_kv_heads,
            intermediate_size=arch.intermediate_size,
            vocab_size=arch.vocab_size,
            max_context_length=arch.max_context_length,
            attention_type=arch.attention_type,
        )