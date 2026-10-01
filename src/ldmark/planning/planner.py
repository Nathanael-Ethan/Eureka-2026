"""
LDMARK Compression Planning Engine

Standalone planner that receives ModelAnalysis + HardwareProfile + user constraints
and produces CompressionPlan with factual feasibility information.

Does NOT rank methods by "best." Returns factual feasibility only.
Does NOT perform model transformation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.ldmark.analysis.models import (
    ArchitectureInfo,
    DType,
    ModelAnalysis,
    ParameterCounts,
    RuntimeMemoryEstimate,
)
from src.ldmark.hardware.memory_budget import MemoryBudget, MemoryBudgetResult, calculate_memory_budget
from src.ldmark.hardware.profile import HardwareProfile


class CompressionMethod(Enum):
    """Compression methods evaluated by the planner."""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass(frozen=True)
class ConstraintConfig:
    """User constraints for compression planning."""
    max_model_storage_bytes: Optional[int] = None
    max_runtime_memory_bytes: Optional[int] = None
    target_precision: Optional[str] = None
    minimum_free_memory_bytes: int = 0
    preferred_backend: Optional[str] = None

    def to_memory_budget(self) -> MemoryBudget:
        return MemoryBudget(
            max_model_storage_bytes=self.max_model_storage_bytes,
            max_runtime_memory_bytes=self.max_runtime_memory_bytes,
            minimum_free_memory_bytes=self.minimum_free_memory_bytes,
            target_precision=self.target_precision,
            preferred_backend=self.preferred_backend,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_storage_bytes": self.max_model_storage_bytes,
            "max_runtime_memory_bytes": self.max_runtime_memory_bytes,
            "target_precision": self.target_precision,
            "minimum_free_memory_bytes": self.minimum_free_memory_bytes,
            "preferred_backend": self.preferred_backend,
        }


@dataclass(frozen=True)
class CompressionStrategy:
    """
    Feasibility information for a single compression method.

    Contains factual calculations only. No quality claims.
    """
    method: str
    bits_per_weight: float
    estimated_storage_bytes: int
    estimated_runtime_weight_memory_bytes: int
    estimated_runtime_total_memory_bytes: int
    compression_ratio: float
    fits_storage_budget: bool
    fits_runtime_budget: bool
    memory_pressure_ratio: float
    warnings: List[str] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    group_size: Optional[int] = None
    scale_dtype: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "bits_per_weight": round(self.bits_per_weight, 4),
            "estimated_storage_bytes": self.estimated_storage_bytes,
            "estimated_runtime_weight_memory_bytes": self.estimated_runtime_weight_memory_bytes,
            "estimated_runtime_total_memory_bytes": self.estimated_runtime_total_memory_bytes,
            "compression_ratio": round(self.compression_ratio, 4),
            "fits_storage_budget": self.fits_storage_budget,
            "fits_runtime_budget": self.fits_runtime_budget,
            "memory_pressure_ratio": round(self.memory_pressure_ratio, 4),
            "warnings": self.warnings,
            "assumptions": self.assumptions,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
        }


@dataclass(frozen=True)
class CompressionPlan:
    """
    Complete compression plan with feasibility for all evaluated methods.

    Does NOT rank methods. Returns factual feasibility information.
    """
    model_id: str
    parameter_count: int
    baseline_dtype: str
    strategies: List[CompressionStrategy]
    target_hardware: str
    constraints: ConstraintConfig
    timestamp: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "parameter_count": self.parameter_count,
            "baseline_dtype": self.baseline_dtype,
            "strategies": [s.to_dict() for s in self.strategies],
            "target_hardware": self.target_hardware,
            "constraints": self.constraints.to_dict(),
            "timestamp": self.timestamp,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class PlanResult:
    """Result of a planning operation."""
    plan: CompressionPlan
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _calculate_groupwise_bits_per_weight(
    weight_bits: float,
    group_size: int,
    scale_bits: int = 16,
) -> float:
    """Calculate effective bits per weight for group-wise quantization."""
    if group_size <= 0:
        return weight_bits
    return (group_size * weight_bits + scale_bits) / group_size


def _calculate_storage_bytes(
    param_count: int,
    bits_per_weight: float,
) -> int:
    """Calculate storage size in bytes."""
    return int(math.ceil(param_count * bits_per_weight / 8.0))


def _calculate_runtime_memory(
    param_count: int,
    weight_bits_per_weight: float,
    architecture: Optional[ArchitectureInfo],
    context_length: int = 2048,
    batch_size: int = 1,
    activation_dtype: DType = DType.FP16,
) -> Dict[str, int]:
    """
    Estimate runtime memory components.

    Returns dict with weights, kv_cache, activations, overhead, total in bytes.
    """
    weights_bytes = int(math.ceil(param_count * weight_bits_per_weight / 8.0))

    kv_cache_bytes = 0
    activations_bytes = 0

    if architecture and architecture.num_layers and architecture.hidden_size:
        num_layers = architecture.num_layers
        hidden_size = architecture.hidden_size
        num_kv_heads = architecture.num_kv_heads or architecture.num_attention_heads or 32
        num_attention_heads = architecture.num_attention_heads or 32
        head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else hidden_size // 32

        kv_elements = 2 * batch_size * num_layers * context_length * num_kv_heads * head_dim
        kv_cache_bytes = int(kv_elements * activation_dtype.bytes_per_element)

        activations_per_layer = batch_size * context_length * hidden_size * 4
        total_activation_elements = activations_per_layer * num_layers
        activations_bytes = int(total_activation_elements * activation_dtype.bytes_per_element)

    overhead_bytes = int((weights_bytes + kv_cache_bytes + activations_bytes) * 0.1)
    total_bytes = weights_bytes + kv_cache_bytes + activations_bytes + overhead_bytes

    return {
        "weights": weights_bytes,
        "kv_cache": kv_cache_bytes,
        "activations": activations_bytes,
        "overhead": overhead_bytes,
        "total": total_bytes,
    }


def _evaluate_strategy(
    method: CompressionMethod,
    param_count: int,
    baseline_bytes: int,
    group_size: int,
    scale_bits: int,
    weight_bits: float,
    architecture: Optional[ArchitectureInfo],
    budget: MemoryBudget,
    available_memory_bytes: Optional[int],
    context_length: int,
    batch_size: int,
) -> CompressionStrategy:
    """Evaluate a single compression strategy for feasibility."""
    bpw = _calculate_groupwise_bits_per_weight(weight_bits, group_size, scale_bits)
    storage_bytes = _calculate_storage_bytes(param_count, bpw)
    compression_ratio = baseline_bytes / storage_bytes if storage_bytes > 0 else 0.0

    runtime = _calculate_runtime_memory(
        param_count, bpw, architecture, context_length, batch_size
    )

    budget_result = calculate_memory_budget(
        model_storage_bytes=storage_bytes,
        weights_bytes=runtime["weights"],
        kv_cache_bytes=runtime["kv_cache"],
        activations_bytes=runtime["activations"],
        overhead_bytes=runtime["overhead"],
        budget=budget,
        available_memory_bytes=available_memory_bytes,
    )

    warnings = list(budget_result.warnings)
    assumptions = dict(budget_result.assumptions)
    assumptions["context_length"] = context_length
    assumptions["batch_size"] = batch_size
    assumptions["group_size"] = group_size
    assumptions["scale_bits_per_group"] = scale_bits

    scale_dtype_str = "fp16" if scale_bits == 16 else f"{scale_bits}-bit"

    return CompressionStrategy(
        method=method.value,
        bits_per_weight=bpw,
        estimated_storage_bytes=storage_bytes,
        estimated_runtime_weight_memory_bytes=runtime["weights"],
        estimated_runtime_total_memory_bytes=runtime["total"],
        compression_ratio=compression_ratio,
        fits_storage_budget=budget_result.fits_storage_budget,
        fits_runtime_budget=budget_result.fits_runtime_budget,
        memory_pressure_ratio=budget_result.memory_pressure_ratio,
        warnings=warnings,
        assumptions=assumptions,
        group_size=group_size,
        scale_dtype=scale_dtype_str,
    )


class CompressionPlanner:
    """
    Standalone compression planning engine.

    Evaluates compression methods for feasibility given model analysis,
    hardware profile, and user constraints. Does NOT rank or recommend.
    """

    DEFAULT_GROUP_SIZE = 128
    DEFAULT_SCALE_BITS = 16
    DEFAULT_CONTEXT_LENGTH = 2048
    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        group_size: int = DEFAULT_GROUP_SIZE,
        scale_bits: int = DEFAULT_SCALE_BITS,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.group_size = group_size
        self.scale_bits = scale_bits
        self.context_length = context_length
        self.batch_size = batch_size

    def plan(
        self,
        analysis: ModelAnalysis,
        hardware: HardwareProfile,
        constraints: Optional[ConstraintConfig] = None,
    ) -> PlanResult:
        """
        Generate a compression plan.

        Args:
            analysis: ModelAnalysis with parameter counts and architecture info.
            hardware: HardwareProfile of the target system.
            constraints: Optional user constraints.

        Returns:
            PlanResult with CompressionPlan or errors.
        """
        from datetime import datetime

        errors = []
        warnings = []
        constraints = constraints or ConstraintConfig()

        param_count = analysis.parameter_counts.total
        if param_count <= 0:
            errors.append("Model has no parameters. Cannot generate compression plan.")
            return PlanResult(
                plan=CompressionPlan(
                    model_id=analysis.model_id,
                    parameter_count=0,
                    baseline_dtype="fp32",
                    strategies=[],
                    target_hardware=hardware.cpu.model,
                    constraints=constraints,
                ),
                success=False,
                errors=errors,
                warnings=warnings,
            )

        baseline_bytes = param_count * 4
        budget = constraints.to_memory_budget()
        available_memory = hardware.system.available_ram_bytes or hardware.system.total_ram_bytes

        architecture = analysis.architecture

        strategies = [
            self._evaluate_fp16(param_count, baseline_bytes, architecture, budget, available_memory),
            self._evaluate_int8(param_count, baseline_bytes, architecture, budget, available_memory),
            self._evaluate_int4(param_count, baseline_bytes, architecture, budget, available_memory),
            self._evaluate_binary(param_count, baseline_bytes, architecture, budget, available_memory),
            self._evaluate_ternary(param_count, baseline_bytes, architecture, budget, available_memory),
        ]

        feasible_methods = [s.method for s in strategies if s.fits_storage_budget and s.fits_runtime_budget]
        if feasible_methods:
            notes = [
                f"Feasible methods under current constraints: {', '.join(feasible_methods)}.",
                "Feasibility is based on storage and memory estimates only.",
                "Quality impact is not estimated at this stage.",
            ]
        else:
            notes = [
                "No methods are feasible under the current constraints.",
                "Consider relaxing storage or memory limits.",
            ]

        plan = CompressionPlan(
            model_id=analysis.model_id,
            parameter_count=param_count,
            baseline_dtype="fp32",
            strategies=strategies,
            target_hardware=hardware.cpu.model,
            constraints=constraints,
            timestamp=datetime.now().isoformat(),
            notes=notes,
        )

        return PlanResult(
            plan=plan,
            success=True,
            errors=errors,
            warnings=warnings,
        )

    def _evaluate_fp16(
        self, param_count: int, baseline_bytes: int,
        architecture: Optional[ArchitectureInfo],
        budget: MemoryBudget, available_memory: Optional[int],
    ) -> CompressionStrategy:
        bpw = 16.0
        storage_bytes = _calculate_storage_bytes(param_count, bpw)
        compression_ratio = baseline_bytes / storage_bytes if storage_bytes > 0 else 0.0
        runtime = _calculate_runtime_memory(param_count, bpw, architecture, self.context_length, self.batch_size)
        budget_result = calculate_memory_budget(
            model_storage_bytes=storage_bytes,
            weights_bytes=runtime["weights"],
            kv_cache_bytes=runtime["kv_cache"],
            activations_bytes=runtime["activations"],
            overhead_bytes=runtime["overhead"],
            budget=budget,
            available_memory_bytes=available_memory,
        )
        return CompressionStrategy(
            method="fp16",
            bits_per_weight=bpw,
            estimated_storage_bytes=storage_bytes,
            estimated_runtime_weight_memory_bytes=runtime["weights"],
            estimated_runtime_total_memory_bytes=runtime["total"],
            compression_ratio=compression_ratio,
            fits_storage_budget=budget_result.fits_storage_budget,
            fits_runtime_budget=budget_result.fits_runtime_budget,
            memory_pressure_ratio=budget_result.memory_pressure_ratio,
            warnings=budget_result.warnings,
            assumptions=budget_result.assumptions,
            group_size=None,
            scale_dtype=None,
        )

    def _evaluate_int8(
        self, param_count: int, baseline_bytes: int,
        architecture: Optional[ArchitectureInfo],
        budget: MemoryBudget, available_memory: Optional[int],
    ) -> CompressionStrategy:
        return _evaluate_strategy(
            CompressionMethod.INT8, param_count, baseline_bytes,
            self.group_size, self.scale_bits, 8.0,
            architecture, budget, available_memory,
            self.context_length, self.batch_size,
        )

    def _evaluate_int4(
        self, param_count: int, baseline_bytes: int,
        architecture: Optional[ArchitectureInfo],
        budget: MemoryBudget, available_memory: Optional[int],
    ) -> CompressionStrategy:
        return _evaluate_strategy(
            CompressionMethod.INT4, param_count, baseline_bytes,
            self.group_size, self.scale_bits, 4.0,
            architecture, budget, available_memory,
            self.context_length, self.batch_size,
        )

    def _evaluate_binary(
        self, param_count: int, baseline_bytes: int,
        architecture: Optional[ArchitectureInfo],
        budget: MemoryBudget, available_memory: Optional[int],
    ) -> CompressionStrategy:
        return _evaluate_strategy(
            CompressionMethod.BINARY, param_count, baseline_bytes,
            self.group_size, self.scale_bits, 1.0,
            architecture, budget, available_memory,
            self.context_length, self.batch_size,
        )

    def _evaluate_ternary(
        self, param_count: int, baseline_bytes: int,
        architecture: Optional[ArchitectureInfo],
        budget: MemoryBudget, available_memory: Optional[int],
    ) -> CompressionStrategy:
        return _evaluate_strategy(
            CompressionMethod.TERNARY, param_count, baseline_bytes,
            self.group_size, self.scale_bits, math.log2(3),
            architecture, budget, available_memory,
            self.context_length, self.batch_size,
        )


def plan_compression(
    analysis: ModelAnalysis,
    hardware: HardwareProfile,
    constraints: Optional[ConstraintConfig] = None,
    group_size: int = 128,
    scale_bits: int = 16,
    context_length: int = 2048,
    batch_size: int = 1,
) -> PlanResult:
    """
    Convenience function to generate a compression plan.

    Args:
        analysis: ModelAnalysis with parameter counts and architecture info.
        hardware: HardwareProfile of the target system.
        constraints: Optional user constraints.
        group_size: Group size for group-wise quantization.
        scale_bits: Bits per scale factor.
        context_length: Context length for runtime memory estimation.
        batch_size: Batch size for runtime memory estimation.

    Returns:
        PlanResult with CompressionPlan or errors.
    """
    planner = CompressionPlanner(
        group_size=group_size,
        scale_bits=scale_bits,
        context_length=context_length,
        batch_size=batch_size,
    )
    return planner.plan(analysis, hardware, constraints)