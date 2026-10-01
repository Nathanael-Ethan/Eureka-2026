"""
LDMARK Compression Planning Engine

Standalone planner that receives ModelAnalysis + HardwareProfile + user constraints
and produces CompressionPlan with factual feasibility information.

Does NOT rank methods by "best." Returns factual feasibility only.
Does NOT perform model transformation.
"""

from .planner import (
    CompressionMethod,
    CompressionPlan,
    CompressionPlanner,
    CompressionStrategy,
    ConstraintConfig,
    PlanResult,
    plan_compression,
)

__all__ = [
    "CompressionMethod",
    "CompressionPlan",
    "CompressionPlanner",
    "CompressionStrategy",
    "ConstraintConfig",
    "PlanResult",
    "plan_compression",
]