from __future__ import annotations

from .models import (
    DType,
    TensorTrainableStatus,
    TensorInfo,
    ParameterCounts,
    ArchitectureInfo,
    StorageEstimate,
    CompressionEstimate,
    RuntimeMemoryEstimate,
    ModelAnalysis,
)

from .tensor import (
    create_tensor_info,
    analyze_numpy_dict,
    analyze_torch_state_dict,
    calculate_tensor_statistics,
    infer_dtype_from_name,
)

from .parameters import (
    count_parameters,
    count_parameters_from_dict,
    count_parameters_from_numpy,
    get_parameter_summary,
    estimate_parameters_from_config,
    compare_parameter_counts,
)

from .architecture import (
    detect_architecture_from_config,
    detect_architecture_from_tensor_names,
    build_architecture_info,
    infer_architecture_from_state_dict,
)

from .storage import (
    estimate_storage_for_dtype,
    estimate_all_storages,
    estimate_storage_from_parameter_counts,
    compare_storage_estimates,
    get_storage_summary,
    estimate_actual_vs_theoretical,
)

from .compression import (
    QuantizationFormat,
    COMMON_FORMATS,
    get_format_by_name,
    estimate_compression_for_format,
    estimate_all_compressions,
    estimate_compression_from_parameter_counts,
    calculate_effective_bps,
    create_custom_format,
    get_compression_summary,
)

from .runtime import (
    KVCacheConfig,
    ActivationConfig,
    estimate_kv_cache,
    estimate_activations,
    estimate_runtime_memory,
    estimate_memory_for_generation,
    get_memory_breakdown,
)

from .analyzer import (
    ModelAnalyzer,
    analyze_model,
    analyze_state_dict,
)

from .io import (
    load_config,
    load_safetensors_metadata,
    load_pytorch_state_dict,
    load_numpy_dict,
    save_analysis_json,
    load_analysis_json,
    create_synthetic_model,
    create_synthetic_state_dict,
)

__all__ = [
    "DType",
    "TensorTrainableStatus",
    "TensorInfo",
    "ParameterCounts",
    "ArchitectureInfo",
    "StorageEstimate",
    "CompressionEstimate",
    "RuntimeMemoryEstimate",
    "ModelAnalysis",
    "create_tensor_info",
    "analyze_numpy_dict",
    "analyze_torch_state_dict",
    "calculate_tensor_statistics",
    "infer_dtype_from_name",
    "count_parameters",
    "count_parameters_from_dict",
    "count_parameters_from_numpy",
    "get_parameter_summary",
    "estimate_parameters_from_config",
    "compare_parameter_counts",
    "detect_architecture_from_config",
    "detect_architecture_from_tensor_names",
    "build_architecture_info",
    "infer_architecture_from_state_dict",
    "estimate_storage_for_dtype",
    "estimate_all_storages",
    "estimate_storage_from_parameter_counts",
    "compare_storage_estimates",
    "get_storage_summary",
    "estimate_actual_vs_theoretical",
    "QuantizationFormat",
    "COMMON_FORMATS",
    "get_format_by_name",
    "estimate_compression_for_format",
    "estimate_all_compressions",
    "estimate_compression_from_parameter_counts",
    "calculate_effective_bps",
    "create_custom_format",
    "get_compression_summary",
    "KVCacheConfig",
    "ActivationConfig",
    "estimate_kv_cache",
    "estimate_activations",
    "estimate_runtime_memory",
    "estimate_memory_for_generation",
    "get_memory_breakdown",
    "ModelAnalyzer",
    "analyze_model",
    "analyze_state_dict",
    "load_config",
    "load_safetensors_metadata",
    "load_pytorch_state_dict",
    "load_numpy_dict",
    "save_analysis_json",
    "load_analysis_json",
    "create_synthetic_model",
    "create_synthetic_state_dict",
]

__version__ = "0.1.0"