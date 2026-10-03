"""
LDMARK Artifact Format Specification

This module defines the data structures for the LDMARK compiled model artifact format.
The format is designed to be:
- Deterministic
- Versioned
- Inspectable without loading all tensors
- Portable
- Extensible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class ArtifactFormatVersion(Enum):
    """Supported artifact format versions."""
    V1 = "ldmark-artifact-1.0"
    
    @classmethod
    def latest(cls) -> "ArtifactFormatVersion":
        return cls.V1
    
    @classmethod
    def from_string(cls, version_str: str) -> "ArtifactFormatVersion":
        for v in cls:
            if v.value == version_str:
                return v
        raise ValueError(f"Unsupported artifact format version: {version_str}")


class CompressionMethod(Enum):
    """Compression methods used in the artifact."""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"
    TERNARY = "ternary"
    PRISMML_Q1_0_G128 = "prismml_q1_0_g128"


class TensorEncoding(Enum):
    """Encoding scheme for tensor data."""
    RAW = "raw"
    GROUPWISE_QUANTIZED = "groupwise_quantized"
    BINARY_PACKED = "binary_packed"
    TERNARY_PACKED = "ternary_packed"


@dataclass
class QuantizationParams:
    """Quantization parameters for a tensor or tensor group."""
    target_bits: int
    group_size: int
    scale_dtype: str = "float16"
    symmetric: bool = True
    zero_point: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "symmetric": self.symmetric,
            "zero_point": self.zero_point,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationParams":
        return cls(
            target_bits=data["target_bits"],
            group_size=data["group_size"],
            scale_dtype=data.get("scale_dtype", "float16"),
            symmetric=data.get("symmetric", True),
            zero_point=data.get("zero_point", False),
        )


@dataclass
class TensorIndexEntry:
    """
    Index entry for a single tensor in the artifact.
    
    Contains enough metadata for a runtime to locate and interpret the tensor
    without loading the full artifact.
    """
    name: str
    original_shape: List[int]
    original_dtype: str
    original_size_bytes: int
    
    # Compression info
    encoding: TensorEncoding
    quantization: Optional[QuantizationParams] = None
    
    # Storage location
    filename: str = ""
    offset: int = 0
    byte_length: int = 0
    
    # Scale data (if separate from main payload)
    scale_filename: str = ""
    scale_offset: int = 0
    scale_byte_length: int = 0
    num_scales: int = 0
    
    # Integrity
    sha256: Optional[str] = None
    scale_sha256: Optional[str] = None
    
    # Error metrics (from validation)
    mae: float = 0.0
    mse: float = 0.0
    max_abs_error: float = 0.0
    relative_error: float = 0.0
    
    # Derived
    bits_per_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "original_shape": self.original_shape,
            "original_dtype": self.original_dtype,
            "original_size_bytes": self.original_size_bytes,
            "encoding": self.encoding.value,
            "quantization": self.quantization.to_dict() if self.quantization else None,
            "filename": self.filename,
            "offset": self.offset,
            "byte_length": self.byte_length,
            "scale_filename": self.scale_filename,
            "scale_offset": self.scale_offset,
            "scale_byte_length": self.scale_byte_length,
            "num_scales": self.num_scales,
            "sha256": self.sha256,
            "scale_sha256": self.scale_sha256,
            "mae": self.mae,
            "mse": self.mse,
            "max_abs_error": self.max_abs_error,
            "relative_error": self.relative_error,
            "bits_per_weight": self.bits_per_weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorIndexEntry":
        return cls(
            name=data["name"],
            original_shape=data["original_shape"],
            original_dtype=data["original_dtype"],
            original_size_bytes=data["original_size_bytes"],
            encoding=TensorEncoding(data["encoding"]),
            quantization=QuantizationParams.from_dict(data["quantization"]) if data.get("quantization") else None,
            filename=data.get("filename", ""),
            offset=data.get("offset", 0),
            byte_length=data.get("byte_length", 0),
            scale_filename=data.get("scale_filename", ""),
            scale_offset=data.get("scale_offset", 0),
            scale_byte_length=data.get("scale_byte_length", 0),
            num_scales=data.get("num_scales", 0),
            sha256=data.get("sha256"),
            scale_sha256=data.get("scale_sha256"),
            mae=data.get("mae", 0.0),
            mse=data.get("mse", 0.0),
            max_abs_error=data.get("max_abs_error", 0.0),
            relative_error=data.get("relative_error", 0.0),
            bits_per_weight=data.get("bits_per_weight", 0.0),
        )


@dataclass
class CompressionInfo:
    """Global compression configuration for the artifact."""
    method: CompressionMethod
    target_bits: int
    group_size: int
    scale_dtype: str = "float16"
    symmetric: bool = True
    per_tensor_overrides: Dict[str, QuantizationParams] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "symmetric": self.symmetric,
            "per_tensor_overrides": {k: v.to_dict() for k, v in self.per_tensor_overrides.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionInfo":
        return cls(
            method=CompressionMethod(data["method"]),
            target_bits=data["target_bits"],
            group_size=data["group_size"],
            scale_dtype=data.get("scale_dtype", "float16"),
            symmetric=data.get("symmetric", True),
            per_tensor_overrides={k: QuantizationParams.from_dict(v) for k, v in data.get("per_tensor_overrides", {}).items()},
        )


@dataclass
class ModelInfo:
    """Model identification and architecture information."""
    model_id: str
    architecture: str
    parameter_count: int
    tensor_count: int
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    vocab_size: Optional[int] = None
    max_context_length: Optional[int] = None
    original_dtype: str = "float16"
    source_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "tensor_count": self.tensor_count,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "vocab_size": self.vocab_size,
            "max_context_length": self.max_context_length,
            "original_dtype": self.original_dtype,
            "source_path": self.source_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(
            model_id=data["model_id"],
            architecture=data["architecture"],
            parameter_count=data["parameter_count"],
            tensor_count=data["tensor_count"],
            num_layers=data.get("num_layers"),
            hidden_size=data.get("hidden_size"),
            intermediate_size=data.get("intermediate_size"),
            num_attention_heads=data.get("num_attention_heads"),
            num_kv_heads=data.get("num_kv_heads"),
            vocab_size=data.get("vocab_size"),
            max_context_length=data.get("max_context_length"),
            original_dtype=data.get("original_dtype", "float16"),
            source_path=data.get("source_path", ""),
        )


@dataclass
class HardwareInfo:
    """Target hardware information."""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_architecture: str = ""
    system_ram_gb: float = 0.0
    gpu_model: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_vendor: Optional[str] = None
    operating_system: str = ""
    python_version: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "cpu_architecture": self.cpu_architecture,
            "system_ram_gb": self.system_ram_gb,
            "gpu_model": self.gpu_model,
            "gpu_vram_gb": self.gpu_vram_gb,
            "gpu_vendor": self.gpu_vendor,
            "operating_system": self.operating_system,
            "python_version": self.python_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HardwareInfo":
        return cls(
            cpu_model=data.get("cpu_model", ""),
            cpu_cores=data.get("cpu_cores", 0),
            cpu_architecture=data.get("cpu_architecture", ""),
            system_ram_gb=data.get("system_ram_gb", 0.0),
            gpu_model=data.get("gpu_model"),
            gpu_vram_gb=data.get("gpu_vram_gb"),
            gpu_vendor=data.get("gpu_vendor"),
            operating_system=data.get("operating_system", ""),
            python_version=data.get("python_version", ""),
        )


@dataclass
class CompilationInfo:
    """Compilation configuration and metadata."""
    ldmark_version: str
    format_version: str
    created_at: str
    optimization_strategy: str = "balanced"
    target_precision: str = "auto"
    max_model_size_gb: Optional[float] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ldmark_version": self.ldmark_version,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "optimization_strategy": self.optimization_strategy,
            "target_precision": self.target_precision,
            "max_model_size_gb": self.max_model_size_gb,
            "config_snapshot": self.config_snapshot,
            "warnings": self.warnings,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompilationInfo":
        return cls(
            ldmark_version=data["ldmark_version"],
            format_version=data["format_version"],
            created_at=data["created_at"],
            optimization_strategy=data.get("optimization_strategy", "balanced"),
            target_precision=data.get("target_precision", "auto"),
            max_model_size_gb=data.get("max_model_size_gb"),
            config_snapshot=data.get("config_snapshot", {}),
            warnings=data.get("warnings", []),
        )


@dataclass
class ValidationInfo:
    """Validation results for the artifact."""
    status: str = "not_run"
    passed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "details": self.details,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationInfo":
        return cls(
            status=data.get("status", "not_run"),
            passed=data.get("passed", False),
            details=data.get("details", {}),
            error=data.get("error"),
        )


@dataclass
class StorageAccounting:
    """Detailed storage accounting for the artifact."""
    raw_tensor_bytes: int = 0
    scale_bytes: int = 0
    metadata_bytes: int = 0
    manifest_bytes: int = 0
    total_artifact_bytes: int = 0
    
    theoretical_bits_per_weight: float = 0.0
    actual_bits_per_weight: float = 0.0
    parameter_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_tensor_bytes": self.raw_tensor_bytes,
            "scale_bytes": self.scale_bytes,
            "metadata_bytes": self.metadata_bytes,
            "manifest_bytes": self.manifest_bytes,
            "total_artifact_bytes": self.total_artifact_bytes,
            "theoretical_bits_per_weight": self.theoretical_bits_per_weight,
            "actual_bits_per_weight": self.actual_bits_per_weight,
            "parameter_count": self.parameter_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageAccounting":
        return cls(
            raw_tensor_bytes=data.get("raw_tensor_bytes", 0),
            scale_bytes=data.get("scale_bytes", 0),
            metadata_bytes=data.get("metadata_bytes", 0),
            manifest_bytes=data.get("manifest_bytes", 0),
            total_artifact_bytes=data.get("total_artifact_bytes", 0),
            theoretical_bits_per_weight=data.get("theoretical_bits_per_weight", 0.0),
            actual_bits_per_weight=data.get("actual_bits_per_weight", 0.0),
            parameter_count=data.get("parameter_count", 0),
        )


@dataclass
class Manifest:
    """
    Artifact manifest - the primary entry point for inspection.
    
    Contains all information needed to understand the artifact without
    loading tensor payloads.
    """
    # Magic and version
    magic: str = "LDMARK"
    format_version: ArtifactFormatVersion = ArtifactFormatVersion.V1
    
    # Model info
    model: Optional[ModelInfo] = None
    
    # Compression info
    compression: Optional[CompressionInfo] = None
    
    # Compilation info
    compilation: Optional[CompilationInfo] = None
    
    # Hardware target
    hardware: Optional[HardwareInfo] = None
    
    # Tensor index
    tensors: List[TensorIndexEntry] = field(default_factory=list)
    
    # Validation
    validation: Optional[ValidationInfo] = None
    
    # Storage accounting
    storage: Optional[StorageAccounting] = None
    
    # Overall checksum of manifest (for integrity)
    manifest_sha256: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic,
            "format_version": self.format_version.value,
            "model": self.model.to_dict() if self.model else None,
            "compression": self.compression.to_dict() if self.compression else None,
            "compilation": self.compilation.to_dict() if self.compilation else None,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "tensors": [t.to_dict() for t in self.tensors],
            "validation": self.validation.to_dict() if self.validation else None,
            "storage": self.storage.to_dict() if self.storage else None,
            "manifest_sha256": self.manifest_sha256,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        return cls(
            magic=data.get("magic", "LDMARK"),
            format_version=ArtifactFormatVersion.from_string(data["format_version"]),
            model=ModelInfo.from_dict(data["model"]) if data.get("model") else None,
            compression=CompressionInfo.from_dict(data["compression"]) if data.get("compression") else None,
            compilation=CompilationInfo.from_dict(data["compilation"]) if data.get("compilation") else None,
            hardware=HardwareInfo.from_dict(data["hardware"]) if data.get("hardware") else None,
            tensors=[TensorIndexEntry.from_dict(t) for t in data.get("tensors", [])],
            validation=ValidationInfo.from_dict(data["validation"]) if data.get("validation") else None,
            storage=StorageAccounting.from_dict(data["storage"]) if data.get("storage") else None,
            manifest_sha256=data.get("manifest_sha256", ""),
        )
    
    def compute_sha256(self) -> str:
        """Compute SHA-256 of manifest (excluding the checksum field itself)."""
        data = self.to_dict()
        data.pop("manifest_sha256", None)
        json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify manifest integrity against stored checksum."""
        if not self.manifest_sha256:
            return False
        return self.compute_sha256() == self.manifest_sha256
    
    def get_tensor_index(self) -> Dict[str, TensorIndexEntry]:
        """Get tensor index as a dictionary for fast lookup."""
        return {t.name: t for t in self.tensors}
    
    def get_total_compressed_bytes(self) -> int:
        """Get total compressed tensor bytes."""
        return sum(t.byte_length for t in self.tensors)
    
    def get_total_scale_bytes(self) -> int:
        """Get total scale bytes."""
        return sum(t.scale_byte_length for t in self.tensors)


@dataclass
class ArtifactFormat:
    """
    Complete artifact format container.
    
    This is the main interface for working with artifacts.
    """
    manifest: Manifest
    artifact_path: str = ""
    
    def get_tensor_names(self) -> List[str]:
        """Get list of all tensor names."""
        return [t.name for t in self.manifest.tensors]
    
    def get_tensor_info(self, name: str) -> Optional[TensorIndexEntry]:
        """Get tensor index entry by name."""
        for t in self.manifest.tensors:
            if t.name == name:
                return t
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.manifest.to_dict()
    
    def to_json(self, indent: int = 2) -> str:
        return self.manifest.to_json(indent)
    
    @classmethod
    def from_manifest(cls, manifest: Manifest, artifact_path: str = "") -> "ArtifactFormat":
        return cls(manifest=manifest, artifact_path=artifact_path)


def get_ldmark_version() -> str:
    """Get LDMARK version."""
    try:
        from ldmark import __version__
        return __version__
    except ImportError:
        return "0.0.0-dev"


def create_compilation_info(
    format_version: ArtifactFormatVersion,
    optimization_strategy: str = "balanced",
    target_precision: str = "auto",
    max_model_size_gb: Optional[float] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> CompilationInfo:
    """Create compilation info with current metadata."""
    return CompilationInfo(
        ldmark_version=get_ldmark_version(),
        format_version=format_version.value,
        created_at=datetime.now().isoformat(),
        optimization_strategy=optimization_strategy,
        target_precision=target_precision,
        max_model_size_gb=max_model_size_gb,
        config_snapshot=config_snapshot or {},
        warnings=warnings or [],
    )