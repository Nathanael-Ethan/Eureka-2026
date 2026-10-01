"""
LDMARK Compiler Configuration

Configuration structures for the model compilation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class TargetHardware(Enum):
    """Target hardware categories for compilation optimization."""
    LAPTOP_CPU = "laptop_cpu"
    LAPTOP_GPU = "laptop_gpu"
    DESKTOP_CPU = "desktop_cpu"
    DESKTOP_GPU = "desktop_gpu"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    SERVER = "server"
    UNKNOWN = "unknown"


class PrecisionTarget(Enum):
    """Target precision for compiled model."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"
    AUTO = "auto"


class OptimizationStrategy(Enum):
    """High-level optimization strategy."""
    SIZE = "size"
    SPEED = "speed"
    BALANCED = "balanced"
    ACCURACY = "accuracy"
    CUSTOM = "custom"


class PipelineStage(Enum):
    """Compilation pipeline stages."""
    LOAD = "load"
    ANALYZE = "analyze"
    PLAN = "plan"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    EXPORT = "export"


class CompressionMethod(Enum):
    """Available compression methods from ldmark.compression."""
    INT8 = "int8"
    INT4 = "int4"
    BINARY = "binary"
    TERNARY = "ternary"
    AUTO = "auto"

    @classmethod
    def from_precision_target(cls, target: "PrecisionTarget") -> "CompressionMethod":
        """Map PrecisionTarget to CompressionMethod."""
        mapping = {
            PrecisionTarget.INT8: cls.INT8,
            PrecisionTarget.INT4: cls.INT4,
            PrecisionTarget.FP32: cls.INT8,  # fallback
            PrecisionTarget.FP16: cls.INT8,  # fallback
            PrecisionTarget.BF16: cls.INT8,  # fallback
        }
        return mapping.get(target, cls.AUTO)

    def is_supported(self) -> bool:
        """Check if this method has an implementation."""
        return self != CompressionMethod.AUTO


@dataclass
class CompressionPlan:
    """
    Structured compression plan selected by the planner.
    
    Contains all information needed to execute compression.
    """
    method: CompressionMethod
    target_bits: int
    group_size: int
    scale_dtype: str  # numpy dtype string
    symmetric: bool
    
    # Estimates
    expected_bits_per_weight: float
    expected_storage_bytes: int
    expected_compression_ratio: float
    
    # Selection rationale
    rationale: str
    warnings: List[str] = field(default_factory=list)
    
    # Per-tensor overrides (optional)
    tensor_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "target_bits": self.target_bits,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "symmetric": self.symmetric,
            "expected_bits_per_weight": self.expected_bits_per_weight,
            "expected_storage_bytes": self.expected_storage_bytes,
            "expected_compression_ratio": self.expected_compression_ratio,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "tensor_overrides": self.tensor_overrides,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionPlan":
        return cls(
            method=CompressionMethod(data["method"]),
            target_bits=data["target_bits"],
            group_size=data["group_size"],
            scale_dtype=data["scale_dtype"],
            symmetric=data["symmetric"],
            expected_bits_per_weight=data["expected_bits_per_weight"],
            expected_storage_bytes=data["expected_storage_bytes"],
            expected_compression_ratio=data["expected_compression_ratio"],
            rationale=data["rationale"],
            warnings=data.get("warnings", []),
            tensor_overrides=data.get("tensor_overrides", {}),
        )


@dataclass
class HardwareProfile:
    """Hardware capability profile for target device."""
    name: str
    cpu_cores: Optional[int] = None
    cpu_arch: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None
    system_memory_gb: Optional[float] = None
    supported_precisions: List[PrecisionTarget] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cpu_cores": self.cpu_cores,
            "cpu_arch": self.cpu_arch,
            "gpu_memory_gb": self.gpu_memory_gb,
            "gpu_compute_capability": self.gpu_compute_capability,
            "system_memory_gb": self.system_memory_gb,
            "supported_precisions": [p.value for p in self.supported_precisions],
            "custom_properties": self.custom_properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareProfile:
        return cls(
            name=data["name"],
            cpu_cores=data.get("cpu_cores"),
            cpu_arch=data.get("cpu_arch"),
            gpu_memory_gb=data.get("gpu_memory_gb"),
            gpu_compute_capability=data.get("gpu_compute_capability"),
            system_memory_gb=data.get("system_memory_gb"),
            supported_precisions=[PrecisionTarget(p) for p in data.get("supported_precisions", [])],
            custom_properties=data.get("custom_properties", {}),
        )


@dataclass
class ValidationConfig:
    """Validation configuration for compilation output."""
    enabled: bool = True
    tolerance_rtol: float = 1e-3
    tolerance_atol: float = 1e-5
    test_inputs: Optional[List[Any]] = None
    max_tokens: int = 100
    compare_logits: bool = True
    compare_hidden_states: bool = False
    custom_validators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tolerance_rtol": self.tolerance_rtol,
            "tolerance_atol": self.tolerance_atol,
            "test_inputs": self.test_inputs,
            "max_tokens": self.max_tokens,
            "compare_logits": self.compare_logits,
            "compare_hidden_states": self.compare_hidden_states,
            "custom_validators": self.custom_validators,
        }


@dataclass
class CompilerConfig:
    """
    Main compiler configuration.

    Represents all settings needed for a compilation run.
    """
    # Input/Output
    input_model_path: Path
    output_path: Path

    # Target
    target_hardware: TargetHardware = TargetHardware.UNKNOWN
    hardware_profile: Optional[HardwareProfile] = None

    # Model constraints
    max_model_size_gb: Optional[float] = None
    target_precision: PrecisionTarget = PrecisionTarget.AUTO

    # Strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    custom_strategy_name: Optional[str] = None

    # Compression planning
    compression_method: CompressionMethod = CompressionMethod.AUTO
    force_group_size: Optional[int] = None
    force_scale_dtype: Optional[str] = None
    force_symmetric: Optional[bool] = None

    # Pipeline control
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: list(PipelineStage))
    skip_stages: List[PipelineStage] = field(default_factory=list)
    stage_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Validation
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Misc
    verbose: bool = False
    dry_run: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_model_path = Path(self.input_model_path)
        self.output_path = Path(self.output_path)

    def get_stage_config(self, stage: PipelineStage) -> Dict[str, Any]:
        """Get configuration for a specific pipeline stage."""
        return self.stage_config.get(stage.value, {})

    def is_stage_enabled(self, stage: PipelineStage) -> bool:
        """Check if a pipeline stage is enabled."""
        return stage in self.enabled_stages and stage not in self.skip_stages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_model_path": str(self.input_model_path),
            "output_path": str(self.output_path),
            "target_hardware": self.target_hardware.value,
            "hardware_profile": self.hardware_profile.to_dict() if self.hardware_profile else None,
            "max_model_size_gb": self.max_model_size_gb,
            "target_precision": self.target_precision.value,
            "optimization_strategy": self.optimization_strategy.value,
            "custom_strategy_name": self.custom_strategy_name,
            "compression_method": self.compression_method.value,
            "force_group_size": self.force_group_size,
            "force_scale_dtype": self.force_scale_dtype,
            "force_symmetric": self.force_symmetric,
            "enabled_stages": [s.value for s in self.enabled_stages],
            "skip_stages": [s.value for s in self.skip_stages],
            "stage_config": self.stage_config,
            "validation": self.validation.to_dict(),
            "verbose": self.verbose,
            "dry_run": self.dry_run,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompilerConfig:
        return cls(
            input_model_path=Path(data["input_model_path"]),
            output_path=Path(data["output_path"]),
            target_hardware=TargetHardware(data.get("target_hardware", "unknown")),
            hardware_profile=HardwareProfile.from_dict(data["hardware_profile"]) if data.get("hardware_profile") else None,
            max_model_size_gb=data.get("max_model_size_gb"),
            target_precision=PrecisionTarget(data.get("target_precision", "auto")),
            optimization_strategy=OptimizationStrategy(data.get("optimization_strategy", "balanced")),
            custom_strategy_name=data.get("custom_strategy_name"),
            compression_method=CompressionMethod(data.get("compression_method", "auto")),
            force_group_size=data.get("force_group_size"),
            force_scale_dtype=data.get("force_scale_dtype"),
            force_symmetric=data.get("force_symmetric"),
            enabled_stages=[PipelineStage(s) for s in data.get("enabled_stages", [s.value for s in PipelineStage])],
            skip_stages=[PipelineStage(s) for s in data.get("skip_stages", [])],
            stage_config=data.get("stage_config", {}),
            validation=ValidationConfig(**data.get("validation", {})),
            verbose=data.get("verbose", False),
            dry_run=data.get("dry_run", False),
            metadata=data.get("metadata", {}),
        )