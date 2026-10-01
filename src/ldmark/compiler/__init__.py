"""
LDMARK Compiler Package

Core compiler orchestration layer for model compilation and compression.
"""

from __future__ import annotations

from .config import (
    CompilerConfig,
    HardwareProfile,
    OptimizationStrategy,
    PipelineStage,
    PrecisionTarget,
    TargetHardware,
    ValidationConfig,
    CompressionMethod,
    CompressionPlan,
)
from .pipeline import Pipeline, PipelineContext, PipelineStageBase, StageResult, StageStatus
from .result import (
    CompilationResult,
    CompilationStatus,
    ModelReference,
    RuntimeMemoryMetrics,
    StorageMetrics,
    ValidationResult,
    ValidationStatus,
    CompressedTensorInfo,
    CompilationArtifact,
)
from .stages import (
    LoadStage,
    AnalyzeStage,
    PlanStage,
    TransformStage,
    ValidateStage,
    ExportStage,
    register_default_stages,
)

__all__ = [
    # Config
    "CompilerConfig",
    "HardwareProfile",
    "OptimizationStrategy",
    "PipelineStage",
    "PrecisionTarget",
    "TargetHardware",
    "ValidationConfig",
    "CompressionMethod",
    "CompressionPlan",
    # Pipeline
    "Pipeline",
    "PipelineContext",
    "PipelineStageBase",
    "StageResult",
    "StageStatus",
    # Result
    "CompilationResult",
    "CompilationStatus",
    "ModelReference",
    "RuntimeMemoryMetrics",
    "StorageMetrics",
    "ValidationResult",
    "ValidationStatus",
    "CompressedTensorInfo",
    "CompilationArtifact",
    # Stages
    "LoadStage",
    "AnalyzeStage",
    "PlanStage",
    "TransformStage",
    "ValidateStage",
    "ExportStage",
    "register_default_stages",
]