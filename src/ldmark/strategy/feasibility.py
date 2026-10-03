"""
LDMARK Strategy Engine - Feasibility Calculator

Calculates factual feasibility of compilation strategies given:
- Model characteristics (parameters, architecture)
- Hardware constraints (RAM, VRAM, backends)
- User requirements (storage limits, runtime limits, precision preferences)

Does NOT perform compression. Does NOT rank strategies. Returns factual feasibility only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ldmark.analysis.models import ArchitectureInfo, DType
from src.ldmark.hardware.profile import HardwareProfile, GPURuntime
from src.ldmark.strategy.models import (
    Backend,
    BackendCompatibility,
    CompilationStrategy,
    ConstraintConfig,
    FeasibilityState,
    HardwareContext,
    ModelContext,
    RuntimeMemoryFeasibility,
    StorageFeasibility,
    WeightRepresentation,
)
from src.ldmark.hardware.memory_budget import MemoryBudget, calculate_memory_budget


DEFAULT_GROUP_SIZE = 128
DEFAULT_SCALE_BITS = 16
DEFAULT_OVERHEAD_FACTOR = 0.1


KNOWN_FORMATS: Dict[WeightRepresentation, Dict[str, Any]] = {
    WeightRepresentation.FP32: {"bits": 32.0, "group_size": None, "scale_bits": 0, "scale_dtype": "none"},
    WeightRepresentation.FP16: {"bits": 16.0, "group_size": None, "scale_bits": 0, "scale_dtype": "none"},
    WeightRepresentation.BF16: {"bits": 16.0, "group_size": None, "scale_bits": 0, "scale_dtype": "none"},
    WeightRepresentation.FP8: {"bits": 8.0, "group_size": 128, "scale_bits": 32, "scale_dtype": "fp32"},
    WeightRepresentation.INT8: {"bits": 8.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.INT4: {"bits": 4.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.INT2: {"bits": 2.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.BINARY: {"bits": 1.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.TERNARY: {"bits": math.log2(3), "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.LDMARK_BINARY: {"bits": 1.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.LDMARK_TERNARY: {"bits": math.log2(3), "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.LDMARK_INT4: {"bits": 4.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
    WeightRepresentation.LDMARK_INT8: {"bits": 8.0, "group_size": DEFAULT_GROUP_SIZE, "scale_bits": DEFAULT_SCALE_BITS, "scale_dtype": "fp16"},
}


BACKEND_KERNEL_AVAILABILITY: Dict[WeightRepresentation, Dict[Backend, bool]] = {
    WeightRepresentation.FP32: {Backend.CPU: True, Backend.CUDA: True, Backend.ROCM: True, Backend.METAL: True},
    WeightRepresentation.FP16: {Backend.CPU: True, Backend.CUDA: True, Backend.ROCM: True, Backend.METAL: True},
    WeightRepresentation.BF16: {Backend.CPU: True, Backend.CUDA: True, Backend.ROCM: True, Backend.METAL: True},
    WeightRepresentation.FP8: {Backend.CPU: False, Backend.CUDA: True, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.INT8: {Backend.CPU: True, Backend.CUDA: True, Backend.ROCM: True, Backend.METAL: True},
    WeightRepresentation.INT4: {Backend.CPU: True, Backend.CUDA: True, Backend.ROCM: True, Backend.METAL: True},
    WeightRepresentation.INT2: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.BINARY: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.TERNARY: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.LDMARK_BINARY: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.LDMARK_TERNARY: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.LDMARK_INT4: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
    WeightRepresentation.LDMARK_INT8: {Backend.CPU: False, Backend.CUDA: False, Backend.ROCM: False, Backend.METAL: False},
}


def calculate_effective_bits_per_weight(
    weight_bits: float,
    group_size: Optional[int],
    scale_bits: int,
) -> float:
    """Calculate effective bits per weight for group-wise quantization."""
    if group_size is None or group_size <= 0:
        return weight_bits
    return (group_size * weight_bits + scale_bits) / group_size


def calculate_storage_bytes(
    param_count: int,
    bits_per_weight: float,
) -> int:
    """Calculate theoretical storage size in bytes."""
    return int(math.ceil(param_count * bits_per_weight / 8.0))


def calculate_actual_storage_bytes(
    param_count: int,
    weight_bits: float,
    group_size: Optional[int],
    scale_bits: int,
) -> int:
    """
    Calculate actual serialized file size including scale factors.
    
    Distinguishes theoretical representation size from actual serialized size.
    
    Args:
        param_count: Number of parameters
        weight_bits: Bits per weight element (e.g., 1 for binary, 1.585 for ternary, 4 for INT4)
        group_size: Group size for group-wise quantization
        scale_bits: Bits per scale factor
    """
    if group_size is None or group_size <= 0:
        return calculate_storage_bytes(param_count, weight_bits)
    
    num_groups = (param_count + group_size - 1) // group_size
    weight_bytes = int(math.ceil(param_count * weight_bits / 8.0))
    scale_bytes = num_groups * (scale_bits // 8)
    
    return weight_bytes + scale_bytes


def calculate_runtime_memory(
    param_count: int,
    bits_per_weight: float,
    architecture: Optional[ModelContext],
    context_length: int,
    batch_size: int,
    activation_dtype: DType = DType.FP16,
    kv_cache_dtype: Optional[DType] = None,
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR,
) -> Dict[str, int]:
    """
    Estimate runtime memory components in bytes.
    
    Returns dict with: weights, kv_cache, activations, overhead, total.
    All values are estimates with explicit assumptions.
    """
    weights_bytes = int(math.ceil(param_count * bits_per_weight / 8.0))
    
    kv_cache_bytes = 0
    activations_bytes = 0
    
    if architecture and architecture.num_layers and architecture.hidden_size:
        num_layers = architecture.num_layers
        hidden_size = architecture.hidden_size
        num_kv_heads = architecture.num_kv_heads or architecture.num_attention_heads or 32
        num_attention_heads = architecture.num_attention_heads or 32
        head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else hidden_size // 32
        
        if architecture.attention_type == "mqa":
            kv_heads = 1
        elif architecture.attention_type == "gqa" and architecture.num_kv_heads == 1:
            kv_heads = 1
        else:
            kv_heads = num_kv_heads
        
        kv_dtype = kv_cache_dtype or activation_dtype
        kv_elements = 2 * batch_size * num_layers * context_length * kv_heads * head_dim
        kv_cache_bytes = int(kv_elements * kv_dtype.bytes_per_element)
        
        activations_per_layer = batch_size * context_length * hidden_size * 4
        total_activation_elements = activations_per_layer * num_layers
        activations_bytes = int(total_activation_elements * activation_dtype.bytes_per_element)
    
    overhead_bytes = int((weights_bytes + kv_cache_bytes + activations_bytes) * overhead_factor)
    total_bytes = weights_bytes + kv_cache_bytes + activations_bytes + overhead_bytes
    
    return {
        "weights": weights_bytes,
        "kv_cache": kv_cache_bytes,
        "activations": activations_bytes,
        "overhead": overhead_bytes,
        "total": total_bytes,
    }


def calculate_kv_cache_bytes(
    architecture: ModelContext,
    context_length: int,
    batch_size: int,
    dtype: DType = DType.FP16,
) -> int:
    """Calculate KV cache size for given context length."""
    if not architecture.num_layers or not architecture.hidden_size:
        return 0
    
    num_layers = architecture.num_layers
    hidden_size = architecture.hidden_size
    num_kv_heads = architecture.num_kv_heads or architecture.num_attention_heads or 32
    num_attention_heads = architecture.num_attention_heads or 32
    head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else hidden_size // 32
    
    if architecture.attention_type == "mqa":
        kv_heads = 1
    elif architecture.attention_type == "gqa" and architecture.num_kv_heads == 1:
        kv_heads = 1
    else:
        kv_heads = num_kv_heads
    
    kv_elements = 2 * batch_size * num_layers * context_length * kv_heads * head_dim
    return int(kv_elements * dtype.bytes_per_element)


def evaluate_storage_feasibility(
    estimated_storage_bytes: int,
    constraint_limit_bytes: Optional[int],
    actual_file_size_bytes: Optional[int] = None,
) -> StorageFeasibility:
    """Evaluate storage feasibility with transparent explanation."""
    estimated_gb = estimated_storage_bytes / (1024 ** 3)
    
    if constraint_limit_bytes is None:
        return StorageFeasibility(
            state=FeasibilityState.UNKNOWN,
            estimated_storage_bytes=estimated_storage_bytes,
            estimated_storage_gb=estimated_gb,
            constraint_limit_bytes=None,
            explanation=f"Estimated weight storage is {estimated_gb:.4f} GB. No storage constraint specified.",
            assumptions={"note": "Theoretical storage estimate. Actual file size may differ due to serialization overhead."}
        )
    
    constraint_gb = constraint_limit_bytes / (1024 ** 3)
    fits = estimated_storage_bytes <= constraint_limit_bytes
    
    if fits:
        explanation = (
            f"Estimated weight storage is {estimated_gb:.4f} GB. "
            f"Requested maximum storage is {constraint_gb:.4f} GB. "
            f"Storage constraint satisfied."
        )
    else:
        explanation = (
            f"Estimated weight storage is {estimated_gb:.4f} GB. "
            f"Requested maximum storage is {constraint_gb:.4f} GB. "
            f"Storage constraint VIOLATED: exceeds by {estimated_gb - constraint_gb:.4f} GB."
        )
    
    assumptions = {
        "note": "Theoretical storage estimate. Actual serialized file size may differ due to padding, metadata, and quantization format overhead."
    }
    if actual_file_size_bytes:
        actual_gb = actual_file_size_bytes / (1024 ** 3)
        assumptions["actual_file_size_bytes"] = actual_file_size_bytes
        assumptions["actual_file_size_gb"] = round(actual_gb, 4)
        if actual_file_size_bytes > constraint_limit_bytes:
            explanation += f" Actual file size ({actual_gb:.4f} GB) also exceeds limit."
    
    return StorageFeasibility(
        state=FeasibilityState.FEASIBLE if fits else FeasibilityState.INFEASIBLE,
        estimated_storage_bytes=estimated_storage_bytes,
        estimated_storage_gb=estimated_gb,
        constraint_limit_bytes=constraint_limit_bytes,
        explanation=explanation,
        assumptions=assumptions,
    )


def evaluate_runtime_feasibility(
    runtime_memory: Dict[str, int],
    constraint_limit_bytes: Optional[int],
    available_memory_bytes: Optional[int],
    context_length: int,
    batch_size: int,
) -> RuntimeMemoryFeasibility:
    """Evaluate runtime memory feasibility with transparent explanation."""
    total_bytes = runtime_memory["total"]
    total_gb = total_bytes / (1024 ** 3)
    weights_bytes = runtime_memory["weights"]
    kv_cache_bytes = runtime_memory["kv_cache"]
    activations_bytes = runtime_memory["activations"]
    overhead_bytes = runtime_memory["overhead"]
    
    assumptions = {
        "context_length": context_length,
        "batch_size": batch_size,
        "note": "Estimates are approximate and architecture-dependent. Actual memory may vary significantly.",
        "weights_gb": round(weights_bytes / (1024 ** 3), 4),
        "kv_cache_gb": round(kv_cache_bytes / (1024 ** 3), 4),
        "activations_gb": round(activations_bytes / (1024 ** 3), 4),
        "overhead_gb": round(overhead_bytes / (1024 ** 3), 4),
    }
    
    if constraint_limit_bytes is None and available_memory_bytes is None:
        return RuntimeMemoryFeasibility(
            state=FeasibilityState.UNKNOWN,
            estimated_weights_bytes=weights_bytes,
            estimated_kv_cache_bytes=kv_cache_bytes,
            estimated_activations_bytes=activations_bytes,
            estimated_overhead_bytes=overhead_bytes,
            estimated_total_bytes=total_bytes,
            estimated_total_gb=total_gb,
            constraint_limit_bytes=None,
            available_memory_bytes=available_memory_bytes,
            memory_pressure_ratio=0.0,
            explanation=f"Estimated runtime memory is {total_gb:.4f} GB. No runtime constraint specified.",
            assumptions=assumptions,
        )
    
    fits_runtime = True
    explanation_parts = [f"Estimated runtime memory is {total_gb:.4f} GB."]
    
    if constraint_limit_bytes is not None:
        constraint_gb = constraint_limit_bytes / (1024 ** 3)
        fits_runtime = total_bytes <= constraint_limit_bytes
        if fits_runtime:
            explanation_parts.append(f"Requested maximum runtime memory is {constraint_gb:.4f} GB. Runtime constraint satisfied.")
        else:
            explanation_parts.append(f"Requested maximum runtime memory is {constraint_gb:.4f} GB. Runtime constraint VIOLATED: exceeds by {total_gb - constraint_gb:.4f} GB.")
    
    pressure_ratio = 0.0
    if available_memory_bytes and available_memory_bytes > 0:
        pressure_ratio = total_bytes / available_memory_bytes
        available_gb = available_memory_bytes / (1024 ** 3)
        explanation_parts.append(f"Available memory is {available_gb:.4f} GB. Memory pressure is {pressure_ratio*100:.1f}%.")
        if pressure_ratio > 0.9:
            explanation_parts.append("WARNING: Runtime memory uses >90% of available memory.")
        if pressure_ratio > 1.0:
            explanation_parts.append("CRITICAL: Runtime memory EXCEEDS available memory.")
            fits_runtime = False
    
    return RuntimeMemoryFeasibility(
        state=FeasibilityState.FEASIBLE if fits_runtime else FeasibilityState.INFEASIBLE,
        estimated_weights_bytes=weights_bytes,
        estimated_kv_cache_bytes=kv_cache_bytes,
        estimated_activations_bytes=activations_bytes,
        estimated_overhead_bytes=overhead_bytes,
        estimated_total_bytes=total_bytes,
        estimated_total_gb=total_gb,
        constraint_limit_bytes=constraint_limit_bytes,
        available_memory_bytes=available_memory_bytes,
        memory_pressure_ratio=pressure_ratio,
        explanation=" ".join(explanation_parts),
        assumptions=assumptions,
    )


def evaluate_backend_compatibility(
    representation: WeightRepresentation,
    target_backends: List[Backend],
    hardware_backends: List[Backend],
) -> List[BackendCompatibility]:
    """Evaluate backend compatibility for a weight representation."""
    results = []
    
    backends_to_check = target_backends if target_backends else hardware_backends
    if not backends_to_check:
        backends_to_check = [Backend.CPU]
    
    kernel_map = BACKEND_KERNEL_AVAILABILITY.get(representation, {})
    
    for backend in backends_to_check:
        has_kernel = kernel_map.get(backend, False)
        
        if has_kernel:
            state = FeasibilityState.FEASIBLE
            explanation = f"{representation.value} has known kernel support on {backend.value}."
        else:
            state = FeasibilityState.INFEASIBLE
            explanation = (
                f"{representation.value} has NO known kernel support on {backend.value}. "
                f"Storage may be feasible but runtime execution is unsupported on this backend."
            )
        
        assumptions = {
            "note": "Kernel availability is based on known implementations. Custom kernels may exist."
        }
        
        results.append(BackendCompatibility(
            backend=backend,
            state=state,
            has_kernel=has_kernel,
            explanation=explanation,
            assumptions=assumptions,
        ))
    
    return results


def build_strategy(
    strategy_id: str,
    representation: WeightRepresentation,
    model_context: ModelContext,
    hardware_context: HardwareContext,
    constraints: ConstraintConfig,
    context_length: int,
    batch_size: int,
) -> CompilationStrategy:
    """Build a complete compilation strategy with feasibility analysis."""
    
    format_info = KNOWN_FORMATS.get(representation, {"bits": 16.0, "group_size": None, "scale_bits": 0, "scale_dtype": "unknown"})
    
    weight_bits = format_info["bits"]
    group_size = format_info["group_size"]
    scale_bits = format_info["scale_bits"]
    scale_dtype = format_info["scale_dtype"]
    
    effective_bpw = calculate_effective_bits_per_weight(weight_bits, group_size, scale_bits)
    theoretical_storage_bytes = calculate_storage_bytes(model_context.parameter_count, effective_bpw)
    actual_storage_bytes = calculate_actual_storage_bytes(model_context.parameter_count, weight_bits, group_size, scale_bits)
    
    runtime_memory = calculate_runtime_memory(
        param_count=model_context.parameter_count,
        bits_per_weight=effective_bpw,
        architecture=model_context,
        context_length=context_length,
        batch_size=batch_size,
    )
    
    storage_feasibility = evaluate_storage_feasibility(
        estimated_storage_bytes=theoretical_storage_bytes,
        constraint_limit_bytes=constraints.max_model_storage_bytes,
        actual_file_size_bytes=actual_storage_bytes if actual_storage_bytes != theoretical_storage_bytes else None,
    )
    
    available_memory = constraints.minimum_available_memory_bytes or hardware_context.available_ram_bytes
    runtime_feasibility = evaluate_runtime_feasibility(
        runtime_memory=runtime_memory,
        constraint_limit_bytes=constraints.max_runtime_memory_bytes,
        available_memory_bytes=available_memory,
        context_length=context_length,
        batch_size=batch_size,
    )
    
    target_backends = [Backend.from_string(b) for b in constraints.target_backends] if constraints.target_backends else hardware_context.supported_backends
    backend_compatibility = evaluate_backend_compatibility(
        representation=representation,
        target_backends=target_backends,
        hardware_backends=hardware_context.supported_backends,
    )
    
    warnings = []
    assumptions = {
        "model_parameter_count": model_context.parameter_count,
        "effective_bits_per_weight": round(effective_bpw, 4),
        "group_size": group_size,
        "scale_bits": scale_bits,
        "scale_dtype": scale_dtype,
        "context_length": context_length,
        "batch_size": batch_size,
        "theoretical_storage_bytes": theoretical_storage_bytes,
        "actual_serialized_storage_bytes": actual_storage_bytes,
        "runtime_weights_bytes": runtime_memory["weights"],
        "runtime_kv_cache_bytes": runtime_memory["kv_cache"],
        "runtime_activations_bytes": runtime_memory["activations"],
        "runtime_overhead_bytes": runtime_memory["overhead"],
        "runtime_total_bytes": runtime_memory["total"],
    }
    
    if not storage_feasibility.state == FeasibilityState.FEASIBLE:
        warnings.append(storage_feasibility.explanation)
    if not runtime_feasibility.state == FeasibilityState.FEASIBLE:
        warnings.append(runtime_feasibility.explanation)
    
    for bc in backend_compatibility:
        if bc.state != FeasibilityState.FEASIBLE:
            warnings.append(bc.explanation)
    
    return CompilationStrategy(
        strategy_id=strategy_id,
        weight_representation=representation,
        bits_per_weight=effective_bpw,
        group_size=group_size,
        scale_dtype=scale_dtype if scale_bits > 0 else None,
        scale_bits=scale_bits,
        storage_feasibility=storage_feasibility,
        runtime_feasibility=runtime_feasibility,
        backend_compatibility=backend_compatibility,
        warnings=warnings,
        assumptions=assumptions,
    )


def get_available_representations(
    preferred_precision: Optional[str] = None,
) -> List[WeightRepresentation]:
    """Get list of weight representations to evaluate, optionally filtered by preference."""
    all_formats = [
        WeightRepresentation.FP32,
        WeightRepresentation.FP16,
        WeightRepresentation.BF16,
        WeightRepresentation.FP8,
        WeightRepresentation.INT8,
        WeightRepresentation.INT4,
        WeightRepresentation.INT2,
        WeightRepresentation.BINARY,
        WeightRepresentation.TERNARY,
        WeightRepresentation.LDMARK_BINARY,
        WeightRepresentation.LDMARK_TERNARY,
        WeightRepresentation.LDMARK_INT4,
        WeightRepresentation.LDMARK_INT8,
    ]
    
    if preferred_precision:
        pref = preferred_precision.lower()
        filtered = [f for f in all_formats if pref in f.value.lower()]
        if filtered:
            return filtered
    
    return all_formats