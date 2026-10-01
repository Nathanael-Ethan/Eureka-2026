"""
LDMARK Compilation Result

Structured result containing all compilation outputs and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ValidationStatus(Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"
    PARTIAL = "partial"


class CompilationStatus(Enum):
    """Overall compilation status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    DRY_RUN = "dry_run"


@dataclass
class ModelReference:
    """Reference to a model (input or output)."""
    path: Path
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    parameter_count: Optional[int] = None
    architecture: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "size_bytes": self.size_bytes,
            "parameter_count": self.parameter_count,
            "architecture": self.architecture,
            "metadata": self.metadata,
        }


@dataclass
class StorageMetrics:
    """Storage-related metrics (model file size on disk)."""
    original_size_bytes: Optional[int] = None
    compressed_size_bytes: Optional[int] = None
    compression_ratio: Optional[float] = None
    original_format: Optional[str] = None
    compressed_format: Optional[str] = None
    bits_per_weight: Optional[float] = None
    scale_overhead_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "original_format": self.original_format,
            "compressed_format": self.compressed_format,
            "bits_per_weight": self.bits_per_weight,
            "scale_overhead_bytes": self.scale_overhead_bytes,
        }


@dataclass
class RuntimeMemoryMetrics:
    """Runtime memory metrics (memory required during inference)."""
    weights_gb: Optional[float] = None
    kv_cache_gb: Optional[float] = None
    activations_gb: Optional[float] = None
    overhead_gb: Optional[float] = None
    total_gb: Optional[float] = None
    assumptions: Dict[str, Any] = field(default_factory=dict)

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
class ValidationResult:
    """Result of output model validation."""
    status: ValidationStatus = ValidationStatus.NOT_RUN
    logits_match: Optional[bool] = None
    hidden_states_match: Optional[bool] = None
    max_logit_diff: Optional[float] = None
    max_hidden_diff: Optional[float] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    test_cases_passed: int = 0
    test_cases_total: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "logits_match": self.logits_match,
            "hidden_states_match": self.hidden_states_match,
            "max_logit_diff": self.max_logit_diff,
            "max_hidden_diff": self.max_hidden_diff,
            "rtol": self.rtol,
            "atol": self.atol,
            "test_cases_passed": self.test_cases_passed,
            "test_cases_total": self.test_cases_total,
            "details": self.details,
            "error": self.error,
        }


@dataclass
class CompressedTensorInfo:
    """Information about a compressed tensor in the compilation artifact."""
    name: str
    original_shape: List[int]
    original_dtype: str
    original_size_bytes: int
    compression_method: str
    target_bits: int
    group_size: int
    scale_dtype: str
    compressed_size_bytes: int
    num_scales: int
    scale_size_bytes: int
    bits_per_weight: float
    mae: float = 0.0
    mse: float = 0.0
    max_abs_error: float = 0.0
    relative_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_shape": self.original_shape,
            "original_dtype": self.original_dtype,
            "original_size_bytes": self.original_size_bytes,
            "compression_method": self.compression_method,
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "compressed_size_bytes": self.compressed_size_bytes,
            "num_scales": self.num_scales,
            "scale_size_bytes": self.scale_size_bytes,
            "bits_per_weight": self.bits_per_weight,
            "mae": self.mae,
            "mse": self.mse,
            "max_abs_error": self.max_abs_error,
            "relative_error": self.relative_error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressedTensorInfo":
        return cls(**data)


@dataclass
class CompilationArtifact:
    """
    Experimental LDMARK compilation artifact.
    
    This is an INTERMEDIATE format - not the final LDMARK binary format.
    Contains all information needed to reconstruct compressed model.
    """
    format_version: str = "ldmark-compilation-0.1"
    is_experimental: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Model metadata
    model_id: str = ""
    source_path: str = ""
    architecture: Optional[str] = None
    parameter_count: int = 0
    
    # Compression configuration
    compression_method: str = ""
    target_bits: int = 0
    group_size: int = 0
    scale_dtype: str = ""
    symmetric: bool = True
    
    # Tensors
    tensors: List[CompressedTensorInfo] = field(default_factory=list)
    
    # Aggregated metrics
    total_original_size_bytes: int = 0
    total_compressed_size_bytes: int = 0
    total_scale_size_bytes: int = 0
    overall_compression_ratio: float = 0.0
    overall_bits_per_weight: float = 0.0
    
    # Validation
    validation_status: str = "not_run"
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "is_experimental": self.is_experimental,
            "created_at": self.created_at,
            "model_id": self.model_id,
            "source_path": self.source_path,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "compression_method": self.compression_method,
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "symmetric": self.symmetric,
            "tensors": [t.to_dict() for t in self.tensors],
            "total_original_size_bytes": self.total_original_size_bytes,
            "total_compressed_size_bytes": self.total_compressed_size_bytes,
            "total_scale_size_bytes": self.total_scale_size_bytes,
            "overall_compression_ratio": self.overall_compression_ratio,
            "overall_bits_per_weight": self.overall_bits_per_weight,
            "validation_status": self.validation_status,
            "validation_details": self.validation_details,
            "warnings": self.warnings,
        }
    
    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompilationArtifact":
        return cls(
            format_version=data.get("format_version", "ldmark-compilation-0.1"),
            is_experimental=data.get("is_experimental", True),
            created_at=data.get("created_at", datetime.now().isoformat()),
            model_id=data.get("model_id", ""),
            source_path=data.get("source_path", ""),
            architecture=data.get("architecture"),
            parameter_count=data.get("parameter_count", 0),
            compression_method=data.get("compression_method", ""),
            target_bits=data.get("target_bits", 0),
            group_size=data.get("group_size", 0),
            scale_dtype=data.get("scale_dtype", ""),
            symmetric=data.get("symmetric", True),
            tensors=[CompressedTensorInfo.from_dict(t) for t in data.get("tensors", [])],
            total_original_size_bytes=data.get("total_original_size_bytes", 0),
            total_compressed_size_bytes=data.get("total_compressed_size_bytes", 0),
            total_scale_size_bytes=data.get("total_scale_size_bytes", 0),
            overall_compression_ratio=data.get("overall_compression_ratio", 0.0),
            overall_bits_per_weight=data.get("overall_bits_per_weight", 0.0),
            validation_status=data.get("validation_status", "not_run"),
            validation_details=data.get("validation_details", {}),
            warnings=data.get("warnings", []),
        )


@dataclass
class CompilationResult:
    """
    Complete result of a compilation run.

    Distinguishes between:
    - Model storage size (file size on disk)
    - Runtime memory (memory required during inference)

    These are NOT the same metric.
    """
    # Status
    status: CompilationStatus = CompilationStatus.FAILED
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Configuration reference
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Models
    input_model: Optional[ModelReference] = None
    output_model: Optional[ModelReference] = None

    # Metrics - STORAGE (file size)
    storage: StorageMetrics = field(default_factory=StorageMetrics)

    # Metrics - RUNTIME MEMORY (inference memory)
    runtime_memory: RuntimeMemoryMetrics = field(default_factory=RuntimeMemoryMetrics)

    # Strategy
    selected_strategy: Optional[str] = None
    strategy_details: Dict[str, Any] = field(default_factory=dict)

    # Compilation artifact
    artifact: Optional[CompilationArtifact] = None

    # Validation
    validation: ValidationResult = field(default_factory=ValidationResult)

    # Pipeline info
    pipeline_stages: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def mark_completed(self, status: CompilationStatus = CompilationStatus.SUCCESS) -> None:
        """Mark compilation as completed."""
        self.completed_at = datetime.now()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.status = status

    def add_error(self, error: str) -> None:
        """Add an error and mark status as failed."""
        self.errors.append(error)
        self.status = CompilationStatus.FAILED

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "config_snapshot": self.config_snapshot,
            "input_model": self.input_model.to_dict() if self.input_model else None,
            "output_model": self.output_model.to_dict() if self.output_model else None,
            "storage": self.storage.to_dict(),
            "runtime_memory": self.runtime_memory.to_dict(),
            "selected_strategy": self.selected_strategy,
            "strategy_details": self.strategy_details,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "validation": self.validation.to_dict(),
            "pipeline_stages": self.pipeline_stages,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompilationResult:
        result = cls(
            status=CompilationStatus(data.get("status", "failed")),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_seconds=data.get("duration_seconds"),
            config_snapshot=data.get("config_snapshot", {}),
            input_model=ModelReference(**data["input_model"]) if data.get("input_model") else None,
            output_model=ModelReference(**data["output_model"]) if data.get("output_model") else None,
            storage=StorageMetrics(**data.get("storage", {})),
            runtime_memory=RuntimeMemoryMetrics(**data.get("runtime_memory", {})),
            selected_strategy=data.get("selected_strategy"),
            strategy_details=data.get("strategy_details", {}),
            artifact=CompilationArtifact.from_dict(data["artifact"]) if data.get("artifact") else None,
            validation=ValidationResult(**data.get("validation", {})),
            pipeline_stages=data.get("pipeline_stages", []),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
        )
        return result