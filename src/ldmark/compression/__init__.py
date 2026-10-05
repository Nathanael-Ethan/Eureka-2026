"""
LDMARK Compression Laboratory

Experimental framework for model compression strategies.
This module provides quantization, dequantization, and error analysis tools.
"""

from .quantize import (
    QuantizationConfig,
    QuantizedTensor,
    QuantizationTarget,
    quantize_groupwise,
    dequantize_groupwise,
    calculate_scale,
    quantize_int8,
    quantize_int4,
    dequantize_int8,
    dequantize_int4,
)
from .metrics import (
    CompressionMetrics,
    calculate_error_metrics,
    calculate_compression_ratio,
)
from .binary_ternary import (
    BinaryConfig,
    TernaryConfig,
    BinaryTensor,
    TernaryTensor,
    calculate_binary_storage,
    calculate_ternary_storage,
    binary_quantize_sign,
    binary_dequantize,
    ternary_quantize_threshold,
    ternary_dequantize,
)
from .prismml_calc import (
    PrismMLConfig,
    calculate_prismml_storage,
    prismml_q1_0_g128_bits_per_weight,
    validate_prismml_claim,
    compare_representations,
)
from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_quantization_experiment,
    run_experiment_suite,
)

__all__ = [
    "QuantizationConfig",
    "QuantizedTensor",
    "QuantizationTarget",
    "quantize_groupwise",
    "dequantize_groupwise",
    "calculate_scale",
    "quantize_int8",
    "quantize_int4",
    "dequantize_int8",
    "dequantize_int4",
    "CompressionMetrics",
    "calculate_error_metrics",
    "calculate_compression_ratio",
    "BinaryConfig",
    "TernaryConfig",
    "BinaryTensor",
    "TernaryTensor",
    "calculate_binary_storage",
    "calculate_ternary_storage",
    "binary_quantize_sign",
    "binary_dequantize",
    "ternary_quantize_threshold",
    "ternary_dequantize",
    "PrismMLConfig",
    "calculate_prismml_storage",
    "prismml_q1_0_g128_bits_per_weight",
    "validate_prismml_claim",
    "compare_representations",
    "ExperimentConfig",
    "ExperimentResult",
    "run_quantization_experiment",
    "run_experiment_suite",
]