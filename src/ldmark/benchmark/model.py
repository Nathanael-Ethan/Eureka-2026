"""
Benchmark data model for LDMARK compression evaluation.

Defines structured records for benchmark experiments with all required fields
for machine-readable analysis and comparison.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import uuid
import platform
import json
import os


@dataclass(frozen=True)
class HardwareMetadata:
    """Hardware and environment metadata for reproducibility."""
    platform: str
    processor: str
    python_version: str
    numpy_version: str
    cpu_count: int
    total_memory_gb: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def capture() -> "HardwareMetadata":
        """Capture current hardware/environment metadata."""
        import sys
        return HardwareMetadata(
            platform=platform.platform(),
            processor=platform.processor(),
            python_version=sys.version,
            numpy_version=np.__version__,
            cpu_count=os.cpu_count() or 0,
            total_memory_gb=0.0,  # Would need psutil for actual memory
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ErrorDistribution:
    """Statistical distribution of reconstruction errors."""
    minimum: float
    median: float
    mean: float
    maximum: float
    std: float
    p25: float
    p75: float
    p95: float
    p99: float
    count: int

    @staticmethod
    def from_errors(errors: np.ndarray) -> "ErrorDistribution":
        """Compute error distribution from array of per-element errors."""
        if len(errors) == 0:
            return ErrorDistribution(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return ErrorDistribution(
            minimum=float(np.min(errors)),
            median=float(np.median(errors)),
            mean=float(np.mean(errors)),
            maximum=float(np.max(errors)),
            std=float(np.std(errors)),
            p25=float(np.percentile(errors, 25)),
            p75=float(np.percentile(errors, 75)),
            p95=float(np.percentile(errors, 95)),
            p99=float(np.percentile(errors, 99)),
            count=len(errors),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingMetrics:
    """Timing metrics for compression/decompression operations."""
    transformation_time_ms: float
    dequantization_time_ms: float
    validation_time_ms: float
    total_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageMetrics:
    """Storage metrics comparing theoretical vs actual."""
    theoretical_bytes: int
    actual_bytes: int
    metadata_bytes: int
    scales_bytes: int
    padding_bytes: int
    tensor_headers_bytes: int
    serialization_overhead_bytes: int
    overhead_ratio: float  # actual / theoretical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TensorBenchmarkResult:
    """Benchmark result for a single tensor."""
    tensor_name: str
    original_shape: tuple
    original_dtype: str
    num_elements: int
    compression_method: str
    target_bits: int
    group_size: int
    scale_dtype: str
    original_bytes: int
    compressed_bytes: int
    actual_bits_per_weight: float
    compression_ratio: float
    error_distribution: ErrorDistribution
    relative_error: float
    timing: TimingMetrics
    storage: StorageMetrics
    random_seed: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["error_distribution"] = self.error_distribution.to_dict()
        d["timing"] = self.timing.to_dict()
        d["storage"] = self.storage.to_dict()
        return d


@dataclass(frozen=True)
class ModelBenchmarkResult:
    """Benchmark result for a complete model (multiple tensors)."""
    model_identifier: str
    parameter_count: int
    tensor_count: int
    tensor_results: List[TensorBenchmarkResult]
    aggregate_compression_ratio: float
    aggregate_bits_per_weight: float
    aggregate_original_bytes: int
    aggregate_compressed_bytes: int
    aggregate_storage: StorageMetrics
    total_timing: TimingMetrics
    random_seed: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tensor_results"] = [tr.to_dict() for tr in self.tensor_results]
        d["aggregate_storage"] = self.aggregate_storage.to_dict()
        d["total_timing"] = self.total_timing.to_dict()
        return d


@dataclass(frozen=True)
class BenchmarkRecord:
    """Complete benchmark record for a compression experiment."""
    experiment_id: str
    model_identifier: str
    parameter_count: int
    tensor_count: int
    source_dtype: str
    compression_method: str
    target_bits: int
    group_size: int
    scale_dtype: str
    original_bytes: int
    compressed_bytes: int
    actual_bits_per_weight: float
    compression_ratio: float
    mae: float
    mse: float
    rmse: float
    max_absolute_error: float
    relative_error: float
    error_distribution: ErrorDistribution
    transformation_time_ms: float
    dequantization_time_ms: float
    validation_time_ms: float
    total_time_ms: float
    theoretical_storage_bytes: int
    actual_storage_bytes: int
    metadata_bytes: int
    scales_bytes: int
    padding_bytes: int
    tensor_headers_bytes: int
    serialization_overhead_bytes: int
    hardware_metadata: Optional[HardwareMetadata] = None
    random_seed: int = 42
    timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["error_distribution"] = self.error_distribution.to_dict()
        if self.hardware_metadata is not None:
            d["hardware_metadata"] = self.hardware_metadata.to_dict()
        else:
            d["hardware_metadata"] = None
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "BenchmarkRecord":
        """Deserialize from dictionary."""
        # Reconstruct nested objects
        d = d.copy()
        d["error_distribution"] = ErrorDistribution(**d["error_distribution"])
        hw = d.get("hardware_metadata")
        if hw is not None:
            d["hardware_metadata"] = HardwareMetadata(**hw)
        else:
            d["hardware_metadata"] = None
        return BenchmarkRecord(**d)

    @staticmethod
    def from_json(json_str: str) -> "BenchmarkRecord":
        """Deserialize from JSON string."""
        return BenchmarkRecord.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    experiment_name: str
    model_identifier: str
    source_dtype: np.dtype
    compression_methods: List[str]  # e.g., ["FP32", "FP16", "INT8", "INT4", "BINARY", "TERNARY"]
    target_bits_list: List[int]
    group_sizes: List[int]
    scale_dtypes: List[np.dtype]
    tensor_shapes: List[tuple]  # For synthetic tensors
    tensor_generators: List[str]  # "random_normal", "random_uniform", "ones", "zeros"
    random_seeds: List[int]
    num_repeats: int = 1
    include_binary: bool = False
    include_ternary: bool = False
    warmup_runs: int = 0
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_dtype"] = np.dtype(self.source_dtype).name
        d["scale_dtypes"] = [np.dtype(dt).name for dt in self.scale_dtypes]
        return d