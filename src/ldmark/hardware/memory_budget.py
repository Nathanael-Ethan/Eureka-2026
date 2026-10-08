"""
Memory budget abstraction for LDMARK compression planning.

Distinguishes MODEL STORAGE from RUNTIME MEMORY.
Runtime budget accounts for weights + KV cache + activations + runtime overhead.
Uses estimated values with explicit uncertainty/assumption metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MemoryBudget:
    """
    Memory budget for a target system.

    Attributes:
        max_model_storage_bytes: Maximum allowed model file/storage size.
        max_runtime_memory_bytes: Maximum allowed runtime memory (weights + KV + activations + overhead).
        minimum_free_memory_bytes: Minimum free memory that must remain after loading.
        target_precision: Optional target precision hint (e.g., "fp16", "int8").
        preferred_backend: Optional preferred compute backend.
    """
    max_model_storage_bytes: Optional[int] = None
    max_runtime_memory_bytes: Optional[int] = None
    minimum_free_memory_bytes: int = 0
    target_precision: Optional[str] = None
    preferred_backend: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_model_storage_bytes": self.max_model_storage_bytes,
            "max_runtime_memory_bytes": self.max_runtime_memory_bytes,
            "minimum_free_memory_bytes": self.minimum_free_memory_bytes,
            "target_precision": self.target_precision,
            "preferred_backend": self.preferred_backend,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryBudget:
        return cls(
            max_model_storage_bytes=data.get("max_model_storage_bytes"),
            max_runtime_memory_bytes=data.get("max_runtime_memory_bytes"),
            minimum_free_memory_bytes=data.get("minimum_free_memory_bytes", 0),
            target_precision=data.get("target_precision"),
            preferred_backend=data.get("preferred_backend"),
        )


@dataclass(frozen=True)
class MemoryBudgetResult:
    """
    Result of checking a memory budget against estimated requirements.

    Attributes:
        model_storage_bytes: Estimated model storage in bytes.
        runtime_memory_bytes: Estimated total runtime memory in bytes.
        weights_bytes: Estimated weight memory in bytes.
        kv_cache_bytes: Estimated KV cache memory in bytes.
        activations_bytes: Estimated activation memory in bytes.
        overhead_bytes: Estimated runtime overhead in bytes.
        fits_storage_budget: Whether model storage fits the storage budget.
        fits_runtime_budget: Whether runtime memory fits the runtime budget.
        memory_pressure_ratio: Runtime memory as a fraction of available memory (0-1+).
        warnings: List of warning messages.
        assumptions: Dict of assumption metadata.
    """
    model_storage_bytes: int
    runtime_memory_bytes: int
    weights_bytes: int
    kv_cache_bytes: int
    activations_bytes: int
    overhead_bytes: int
    fits_storage_budget: bool
    fits_runtime_budget: bool
    memory_pressure_ratio: float
    warnings: List[str] = field(default_factory=list)
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_storage_bytes": self.model_storage_bytes,
            "runtime_memory_bytes": self.runtime_memory_bytes,
            "weights_bytes": self.weights_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activations_bytes": self.activations_bytes,
            "overhead_bytes": self.overhead_bytes,
            "fits_storage_budget": self.fits_storage_budget,
            "fits_runtime_budget": self.fits_runtime_budget,
            "memory_pressure_ratio": round(self.memory_pressure_ratio, 4),
            "warnings": self.warnings,
            "assumptions": self.assumptions,
        }


def calculate_memory_budget(
    model_storage_bytes: int,
    weights_bytes: int,
    kv_cache_bytes: int = 0,
    activations_bytes: int = 0,
    overhead_bytes: int = 0,
    budget: Optional[MemoryBudget] = None,
    available_memory_bytes: Optional[int] = None,
) -> MemoryBudgetResult:
    """
    Calculate memory budget feasibility.

    Args:
        model_storage_bytes: Estimated model storage size.
        weights_bytes: Estimated weight memory at runtime.
        kv_cache_bytes: Estimated KV cache memory.
        activations_bytes: Estimated activation memory.
        overhead_bytes: Estimated runtime overhead.
        budget: Optional memory budget constraints.
        available_memory_bytes: Optional total available memory for pressure calculation.

    Returns:
        MemoryBudgetResult with feasibility information.
    """
    runtime_memory_bytes = weights_bytes + kv_cache_bytes + activations_bytes + overhead_bytes

    warnings = []
    assumptions = {
        "kv_cache_bytes": kv_cache_bytes,
        "activations_bytes": activations_bytes,
        "overhead_bytes": overhead_bytes,
        "note": "Estimates are approximate and architecture-dependent.",
    }

    fits_storage = True
    fits_runtime = True
    pressure_ratio = 0.0

    if budget:
        if budget.max_model_storage_bytes is not None:
            fits_storage = model_storage_bytes <= budget.max_model_storage_bytes
            if not fits_storage:
                warnings.append(
                    f"Estimated model storage ({model_storage_bytes:,} bytes) exceeds "
                    f"the requested limit ({budget.max_model_storage_bytes:,} bytes)."
                )

        if budget.max_runtime_memory_bytes is not None:
            fits_runtime = runtime_memory_bytes <= budget.max_runtime_memory_bytes
            if not fits_runtime:
                warnings.append(
                    f"Estimated runtime memory ({runtime_memory_bytes:,} bytes) exceeds "
                    f"the requested limit ({budget.max_runtime_memory_bytes:,} bytes)."
                )

        if available_memory_bytes and available_memory_bytes > 0:
            pressure_ratio = runtime_memory_bytes / available_memory_bytes
            if pressure_ratio > 0.9:
                warnings.append(
                    f"Runtime memory uses {pressure_ratio*100:.1f}% of available memory."
                )

        if budget.minimum_free_memory_bytes > 0 and available_memory_bytes:
            remaining = available_memory_bytes - runtime_memory_bytes
            if remaining < budget.minimum_free_memory_bytes:
                warnings.append(
                    f"Remaining memory ({remaining:,} bytes) is below the minimum "
                    f"free memory requirement ({budget.minimum_free_memory_bytes:,} bytes)."
                )
    else:
        if available_memory_bytes and available_memory_bytes > 0:
            pressure_ratio = runtime_memory_bytes / available_memory_bytes

    return MemoryBudgetResult(
        model_storage_bytes=model_storage_bytes,
        runtime_memory_bytes=runtime_memory_bytes,
        weights_bytes=weights_bytes,
        kv_cache_bytes=kv_cache_bytes,
        activations_bytes=activations_bytes,
        overhead_bytes=overhead_bytes,
        fits_storage_budget=fits_storage,
        fits_runtime_budget=fits_runtime,
        memory_pressure_ratio=pressure_ratio,
        warnings=warnings,
        assumptions=assumptions,
    )


# --- Predefined laptop budgets (M3) ---
# Keep STORAGE budget separate from RUNTIME budget. Runtime covers weights
# (decompressed) + KV cache + activations + overhead.

def laptop_budget_4gb(
    reserve_free_bytes: int = 1 * 1024 ** 3,
    overhead_bytes: int = 256 * 1024 ** 2,
) -> MemoryBudget:
    """4GB laptop: ~2GB usable runtime budget, 1GB kept free."""
    return MemoryBudget(
        max_model_storage_bytes=2 * 1024 ** 3,
        max_runtime_memory_bytes=2 * 1024 ** 3,
        minimum_free_memory_bytes=reserve_free_bytes,
        target_precision="int8",
        preferred_backend="cpu",
    )


def laptop_budget_8gb(
    reserve_free_bytes: int = 1 * 1024 ** 3,
    overhead_bytes: int = 256 * 1024 ** 2,
) -> MemoryBudget:
    """8GB laptop (M1 baseline): ~4GB usable runtime budget, 1GB kept free."""
    return MemoryBudget(
        max_model_storage_bytes=4 * 1024 ** 3,
        max_runtime_memory_bytes=4 * 1024 ** 3,
        minimum_free_memory_bytes=reserve_free_bytes,
        target_precision="int8",
        preferred_backend="cpu",
    )


def check_runtime_against_budget(
    estimated_runtime_bytes: int,
    budget: MemoryBudget,
    label: str = "plan",
) -> MemoryBudgetResult:
    """Enforce a runtime plan against a laptop budget.

    Returns the MemoryBudgetResult; callers should reject when
    fits_runtime_budget is False. The rejection message names the overage.
    """
    result = calculate_memory_budget(
        model_storage_bytes=0,
        weights_bytes=estimated_runtime_bytes,
        budget=budget,
        available_memory_bytes=budget.max_runtime_memory_bytes,
    )
    if not result.fits_runtime_budget:
        over = estimated_runtime_bytes - (budget.max_runtime_memory_bytes or 0)
        result.warnings.append(
            f"REJECTED: {label} needs {estimated_runtime_bytes:,} bytes runtime "
            f"({estimated_runtime_bytes / 1024**3:.3f} GB), exceeding budget "
            f"{budget.max_runtime_memory_bytes:,} bytes by {over:,} bytes. "
            f"Reduce decompressed weights, context length, or batch size."
        )
    return result