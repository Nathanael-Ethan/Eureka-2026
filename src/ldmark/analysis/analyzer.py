from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import os

from .models import (
    ModelAnalysis,
    ParameterCounts,
    TensorInfo,
    ArchitectureInfo,
    StorageEstimate,
    CompressionEstimate,
    RuntimeMemoryEstimate,
    DType,
    TensorTrainableStatus,
)
from .tensor import (
    create_tensor_info,
    analyze_numpy_dict,
    analyze_torch_state_dict,
    calculate_tensor_statistics,
    infer_dtype_from_name,
)
from .parameters import count_parameters, estimate_parameters_from_config, compare_parameter_counts
from .architecture import build_architecture_info, infer_architecture_from_state_dict
from .storage import estimate_all_storages, estimate_storage_from_parameter_counts, estimate_actual_vs_theoretical
from .compression import estimate_all_compressions, estimate_compression_from_parameter_counts, COMMON_FORMATS
from .runtime import estimate_runtime_memory, estimate_memory_for_generation, get_memory_breakdown


class ModelAnalyzer:
    def __init__(self, model_id: str = "unknown", source_path: str = ""):
        self.model_id = model_id
        self.source_path = source_path
        self.tensors: List[TensorInfo] = []
        self.parameter_counts: Optional[ParameterCounts] = None
        self.architecture: Optional[ArchitectureInfo] = None
        self.config: Dict[str, Any] = {}
        self.actual_file_size_bytes: Optional[int] = None
        self.warnings: List[str] = []

    def analyze_tensors(
        self,
        tensors: Union[Dict[str, Any], List[TensorInfo]],
        trainable_names: Optional[set] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelAnalysis:
        self.config = config or {}

        if isinstance(tensors, list):
            self.tensors = tensors
        elif isinstance(tensors, dict):
            if all(isinstance(v, TensorInfo) for v in tensors.values()):
                self.tensors = list(tensors.values())
            else:
                self.tensors = analyze_torch_state_dict(tensors, trainable_names)

        self.parameter_counts = count_parameters(self.tensors)
        self.architecture = build_architecture_info(self.config, [t.name for t in self.tensors])

        if self.config and not self.architecture.model_architecture:
            estimated = estimate_parameters_from_config(self.config)
            if estimated:
                comparison = compare_parameter_counts(self.parameter_counts, estimated)
                if not comparison["match"]:
                    self.warnings.append(
                        f"Parameter count mismatch: actual={self.parameter_counts.total}, "
                        f"estimated={estimated} ({comparison['percent_difference']:.1f}% diff)"
                    )

        return self.build_analysis()

    def analyze_numpy_tensors(
        self,
        tensors: Dict[str, Any],
        trainable_names: Optional[set] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelAnalysis:
        self.config = config or {}
        self.tensors = analyze_numpy_dict(tensors, trainable_names)
        self.parameter_counts = count_parameters(self.tensors)
        self.architecture = build_architecture_info(self.config, [t.name for t in self.tensors])
        return self.build_analysis()

    def analyze_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        trainable_names: Optional[set] = None,
    ) -> ModelAnalysis:
        return self.analyze_tensors(state_dict, trainable_names, config)

    def analyze_from_file(
        self,
        file_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelAnalysis:
        path = Path(file_path)
        self.source_path = str(path)
        self.model_id = path.stem
        self.actual_file_size_bytes = path.stat().st_size if path.exists() else None

        if path.suffix in [".bin", ".pt", ".pth", ".safetensors"]:
            self.warnings.append(
                f"File format {path.suffix} detected. Use load_state_dict() to load tensors first, "
                "then pass to analyze_tensors(). This analyzer does not load model files directly."
            )
            self.tensors = []
            self.parameter_counts = ParameterCounts()
            self.architecture = ArchitectureInfo()
        elif path.suffix == ".json":
            with open(path, "r") as f:
                self.config = json.load(f)
            self.tensors = []
            self.parameter_counts = ParameterCounts()
            self.architecture = build_architecture_info(self.config, [])
        else:
            self.warnings.append(f"Unsupported file format: {path.suffix}")

        return self.build_analysis()

    def set_actual_file_size(self, size_bytes: int):
        self.actual_file_size_bytes = size_bytes

    def build_analysis(self) -> ModelAnalysis:
        if not self.parameter_counts:
            self.parameter_counts = count_parameters(self.tensors)
        if not self.architecture:
            self.architecture = build_architecture_info(self.config, [t.name for t in self.tensors])

        storage_estimates = estimate_storage_from_parameter_counts(self.parameter_counts)
        compression_estimates = estimate_compression_from_parameter_counts(self.parameter_counts)

        runtime_memory = None
        if self.architecture and self.architecture.num_layers and self.architecture.hidden_size:
            weight_dtype = self._infer_weight_dtype()
            runtime_memory = estimate_runtime_memory(
                param_count=self.parameter_counts.total,
                weight_dtype=weight_dtype,
                architecture=self.architecture,
            )

        analysis = ModelAnalysis(
            model_id=self.model_id,
            source_path=self.source_path,
            parameter_counts=self.parameter_counts,
            tensors=self.tensors,
            architecture=self.architecture,
            storage_estimates=storage_estimates,
            compression_estimates=compression_estimates,
            runtime_memory=runtime_memory,
            actual_file_size_bytes=self.actual_file_size_bytes,
            warnings=self.warnings.copy(),
        )

        if self.actual_file_size_bytes and storage_estimates:
            baseline = storage_estimates[0]
            comparison = estimate_actual_vs_theoretical(self.actual_file_size_bytes, baseline.estimated_bytes)
            analysis.warnings.append(
                f"File size vs theoretical: {comparison['interpretation']} "
                f"(ratio: {comparison['ratio']:.2f}x)"
            )

        return analysis

    def _infer_weight_dtype(self) -> DType:
        if self.parameter_counts.by_dtype:
            return max(self.parameter_counts.by_dtype.items(), key=lambda x: x[1])[0]
        for tensor in self.tensors:
            if tensor.num_elements > 0:
                return tensor.dtype
        return DType.FP16

    def get_summary(self) -> Dict[str, Any]:
        if not self.parameter_counts:
            return {"error": "No analysis performed yet"}

        weight_dtype = self._infer_weight_dtype()

        return {
            "model_id": self.model_id,
            "source_path": self.source_path,
            "total_parameters": self.parameter_counts.total,
            "trainable_parameters": self.parameter_counts.trainable,
            "architecture": self.architecture.model_architecture,
            "num_layers": self.architecture.num_layers,
            "hidden_size": self.architecture.hidden_size,
            "weight_dtype": weight_dtype.value,
            "estimated_fp16_size_gb": next(
                (e.estimated_gb for e in estimate_all_storages(self.parameter_counts.total) if e.dtype == DType.FP16),
                0.0,
            ),
            "estimated_int4_size_gb": next(
                (e.estimated_gb for e in estimate_all_storages(self.parameter_counts.total) if e.dtype == DType.INT4),
                0.0,
            ),
            "estimated_int8_size_gb": next(
                (e.estimated_gb for e in estimate_all_storages(self.parameter_counts.total) if e.dtype == DType.INT8),
                0.0,
            ),
            "num_tensors": len(self.tensors),
            "actual_file_size_gb": round(self.actual_file_size_bytes / (1024 ** 3), 4) if self.actual_file_size_bytes else None,
            "warnings": self.warnings,
        }


def analyze_model(
    tensors: Union[Dict[str, Any], List[TensorInfo]],
    model_id: str = "unknown",
    source_path: str = "",
    config: Optional[Dict[str, Any]] = None,
    trainable_names: Optional[set] = None,
) -> ModelAnalysis:
    analyzer = ModelAnalyzer(model_id, source_path)
    return analyzer.analyze_tensors(tensors, trainable_names, config)


def analyze_state_dict(
    state_dict: Dict[str, Any],
    model_id: str = "unknown",
    source_path: str = "",
    config: Optional[Dict[str, Any]] = None,
    trainable_names: Optional[set] = None,
) -> ModelAnalysis:
    analyzer = ModelAnalyzer(model_id, source_path)
    return analyzer.analyze_from_state_dict(state_dict, config, trainable_names)