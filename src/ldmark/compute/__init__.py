"""
LDMARK Low-Bit Computation Engine

Direct computation on compressed weight representations.
"""

from __future__ import annotations

from .abstraction import (
    LowBitMatrix,
    LowBitMatmul,
    ComputationConfig,
    ComputeResult,
    MemoryStats,
    TimingStats,
    ComputePrecision,
    create_reference_result,
    compare_results,
)

__all__ = [
    "LowBitMatrix",
    "LowBitMatmul",
    "ComputationConfig",
    "ComputeResult",
    "MemoryStats",
    "TimingStats",
    "ComputePrecision",
    "create_reference_result",
    "compare_results",
]