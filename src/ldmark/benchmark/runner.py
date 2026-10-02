"""
Benchmark runner for LDMARK compression laboratory.

Executes tensor-level and model-level benchmarks across compression methods.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

from .model import (
    BenchmarkRecord,
    BenchmarkConfig,
    TensorBenchmarkResult,
    ModelBenchmarkResult,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
    HardwareMetadata,
)
from .storage import (
    calculate_theoretical_storage,
    calculate_actual_storage,
    StorageValidationResult,
)

# Import compression functions
from ..compression.quantize import (
    QuantizationConfig,
    QuantizedTensor,
    QuantizationTarget,
    quantize_groupwise,
    dequantize_groupwise,
    calculate_scale,
)
from ..compression.metrics import (
    calculate_error_metrics,
    calculate_compression_ratio,
    calculate_storage_size,
)
from ..compression.binary_ternary import (
    BinaryConfig,
    TernaryConfig,
    BinaryTensor,
    TernaryTensor,
    binary_quantize_sign,
    binary_dequantize,
    ternary_quantize_threshold,
    ternary_dequantize,
    calculate_binary_storage,
    calculate_ternary_storage,
)
from ..compression.experiment import generate_test_tensor


def _measure_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def _compute_error_distribution(original: np.ndarray, reconstructed: np.ndarray) -> ErrorDistribution:
    """Compute per-element error distribution."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_errors = np.abs(diff).flatten()
    return ErrorDistribution.from_errors(abs_errors)


def _quantize_fp32(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    """FP32 is no-op, just measure conversion time."""
    start = time.perf_counter()
    result = tensor.astype(np.float32)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def _quantize_fp16(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    """FP16 conversion."""
    start = time.perf_counter()
    result = tensor.astype(np.float16)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def _quantize_int8(tensor: np.ndarray, group_size: int, scale_dtype: np.dtype) -> Tuple[QuantizedTensor, float]:
    """INT8 group-wise quantization."""
    config = QuantizationConfig(
        target_bits=QuantizationTarget.INT8,
        group_size=group_size,
        scale_dtype=scale_dtype,
        symmetric=True,
    )
    return _measure_time(quantize_groupwise, tensor, config)


def _dequantize_int8(qtensor: QuantizedTensor) -> Tuple[np.ndarray, float]:
    """INT8 dequantization."""
    return _measure_time(dequantize_groupwise, qtensor)


def _quantize_int4(tensor: np.ndarray, group_size: int, scale_dtype: np.dtype) -> Tuple[QuantizedTensor, float]:
    """INT4 group-wise quantization."""
    config = QuantizationConfig(
        target_bits=QuantizationTarget.INT4,
        group_size=group_size,
        scale_dtype=scale_dtype,
        symmetric=True,
    )
    return _measure_time(quantize_groupwise, tensor, config)


def _dequantize_int4(qtensor: QuantizedTensor) -> Tuple[np.ndarray, float]:
    """INT4 dequantization."""
    return _measure_time(dequantize_groupwise, qtensor)


def _quantize_binary(tensor: np.ndarray, group_size: int, scale_dtype: np.dtype) -> Tuple[BinaryTensor, float]:
    """Binary quantization (research only)."""
    config = BinaryConfig(group_size=group_size, scale_dtype=scale_dtype)
    return _measure_time(binary_quantize_sign, tensor, config)


def _dequantize_binary(btensor: BinaryTensor) -> Tuple[np.ndarray, float]:
    """Binary dequantization."""
    return _measure_time(binary_dequantize, btensor)


def _quantize_ternary(tensor: np.ndarray, group_size: int, scale_dtype: np.dtype, threshold: float = 0.05) -> Tuple[TernaryTensor, float]:
    """Ternary quantization (research only)."""
    config = TernaryConfig(group_size=group_size, scale_dtype=scale_dtype)
    return _measure_time(ternary_quantize_threshold, tensor, config, threshold)


def _dequantize_ternary(ttensor: TernaryTensor) -> Tuple[np.ndarray, float]:
    """Ternary dequantization."""
    return _measure_time(ternary_dequantize, ttensor)


def run_tensor_benchmark(
    tensor: np.ndarray,
    tensor_name: str,
    compression_method: str,
    target_bits: int,
    group_size: int,
    scale_dtype: np.dtype,
    random_seed: int,
) -> TensorBenchmarkResult:
    """
    Run a benchmark on a single tensor with specified compression method.
    
    Args:
        tensor: Input tensor to compress
        tensor_name: Identifier for the tensor
        compression_method: One of "FP32", "FP16", "INT8", "INT4", "BINARY", "TERNARY"
        target_bits: Target bits per weight (for INT8=8, INT4=4, BINARY=1, TERNARY=~1.585)
        group_size: Group size for grouped quantization
        scale_dtype: Data type for scale factors
        random_seed: Random seed for reproducibility
        
    Returns:
        TensorBenchmarkResult with all metrics
    """
    original_shape = tensor.shape
    original_dtype = str(tensor.dtype)
    num_elements = tensor.size
    original_bytes = tensor.nbytes
    
    # Quantize
    if compression_method == "FP32":
        quantized_tensor, transform_time = _quantize_fp32(tensor)
        dequantized_tensor, dequant_time = _measure_time(lambda x: x.astype(np.float32), quantized_tensor)
        compressed_bytes = quantized_tensor.nbytes
        scales_bytes = 0
        metadata_bytes = 0
        padding_bytes = 0
        tensor_headers_bytes = 0
        serialization_overhead_bytes = 0
        
    elif compression_method == "FP16":
        quantized_tensor, transform_time = _quantize_fp16(tensor)
        dequantized_tensor, dequant_time = _measure_time(lambda x: x.astype(np.float32), quantized_tensor)
        compressed_bytes = quantized_tensor.nbytes
        scales_bytes = 0
        metadata_bytes = 0
        padding_bytes = 0
        tensor_headers_bytes = 0
        serialization_overhead_bytes = 0
        
    elif compression_method == "INT8":
        qtensor, transform_time = _quantize_int8(tensor, group_size, scale_dtype)
        dequantized_tensor, dequant_time = _dequantize_int8(qtensor)
        compressed_bytes = qtensor.data.nbytes
        scales_bytes = qtensor.scales.nbytes
        metadata_bytes = 0  # No additional metadata in current format
        padding_bytes = 0
        tensor_headers_bytes = 0
        serialization_overhead_bytes = 0
        
    elif compression_method == "INT4":
        qtensor, transform_time = _quantize_int4(tensor, group_size, scale_dtype)
        dequantized_tensor, dequant_time = _dequantize_int4(qtensor)
        compressed_bytes = qtensor.data.nbytes
        scales_bytes = qtensor.scales.nbytes
        metadata_bytes = 0
        padding_bytes = 0
        tensor_headers_bytes = 0
        serialization_overhead_bytes = 0
        
    elif compression_method == "BINARY":
        btensor, transform_time = _quantize_binary(tensor, group_size, scale_dtype)
        dequantized_tensor, dequant_time = _dequantize_binary(btensor)
        compressed_bytes = btensor.packed_data.nbytes
        scales_bytes = btensor.scales.nbytes
        metadata_bytes = 0
        padding_bytes = 0
        tensor_headers_bytes = 0
        serialization_overhead_bytes = 0
        
    elif compression_method == "TERNARY":
        ttensor, transform_time = _quantize_ternary(tensor, group_size, scale_dtype)
        dequantized_tensor, dequant_time = _dequantize_ternary(ttensor)
        compressed_bytes = ttensor.packed_data.nbytes
        scales_bytes = ttensor.scales.nbytes
        metadata_bytes = 0
        padding_bytes = 0
        tensor_headers_bytes = 0
        serialization_overhead_bytes = 0
        
    else:
        raise ValueError(f"Unknown compression method: {compression_method}")
    
    # Validation timing
    validation_start = time.perf_counter()
    error_dist = _compute_error_distribution(tensor, dequantized_tensor)
    mae, mse, max_err, rel_err = calculate_error_metrics(tensor, dequantized_tensor)
    validation_time = (time.perf_counter() - validation_start) * 1000
    
    # Compression ratio and bits per weight
    compression_ratio = calculate_compression_ratio(original_bytes, compressed_bytes + scales_bytes)
    total_compressed_bits = (compressed_bytes + scales_bytes) * 8
    actual_bits_per_weight = total_compressed_bits / num_elements
    
    # Storage metrics
    theoretical_bytes = calculate_theoretical_storage(
        num_elements, target_bits, group_size, np.dtype(scale_dtype).itemsize * 8
    )
    storage = StorageMetrics(
        theoretical_bytes=theoretical_bytes,
        actual_bytes=compressed_bytes + scales_bytes,
        metadata_bytes=metadata_bytes,
        scales_bytes=scales_bytes,
        padding_bytes=padding_bytes,
        tensor_headers_bytes=tensor_headers_bytes,
        serialization_overhead_bytes=serialization_overhead_bytes,
        overhead_ratio=(compressed_bytes + scales_bytes) / theoretical_bytes if theoretical_bytes > 0 else 0.0,
    )
    
    timing = TimingMetrics(
        transformation_time_ms=transform_time,
        dequantization_time_ms=dequant_time,
        validation_time_ms=validation_time,
        total_time_ms=transform_time + dequant_time + validation_time,
    )
    
    return TensorBenchmarkResult(
        tensor_name=tensor_name,
        original_shape=original_shape,
        original_dtype=original_dtype,
        num_elements=num_elements,
        compression_method=compression_method,
        target_bits=target_bits,
        group_size=group_size,
        scale_dtype=str(scale_dtype),
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes + scales_bytes,
        actual_bits_per_weight=actual_bits_per_weight,
        compression_ratio=compression_ratio,
        error_distribution=error_dist,
        relative_error=rel_err,
        timing=timing,
        storage=storage,
        random_seed=random_seed,
    )


def run_model_benchmark(
    tensors: Dict[str, np.ndarray],
    model_identifier: str,
    compression_method: str,
    target_bits: int,
    group_size: int,
    scale_dtype: np.dtype,
    random_seed: int,
) -> ModelBenchmarkResult:
    """
    Run benchmark on a complete model (dictionary of tensors).
    
    Args:
        tensors: Dictionary mapping tensor names to tensor arrays
        model_identifier: Model identifier string
        compression_method: Compression method to apply
        target_bits: Target bits per weight
        group_size: Group size for grouped quantization
        scale_dtype: Scale data type
        random_seed: Random seed
        
    Returns:
        ModelBenchmarkResult with aggregate metrics
    """
    tensor_results = []
    total_original_bytes = 0
    total_compressed_bytes = 0
    total_elements = 0
    total_transform_time = 0.0
    total_dequant_time = 0.0
    total_validation_time = 0.0
    
    for name, tensor in tensors.items():
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name=name,
            compression_method=compression_method,
            target_bits=target_bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            random_seed=random_seed,
        )
        tensor_results.append(result)
        total_original_bytes += result.original_bytes
        total_compressed_bytes += result.compressed_bytes
        total_elements += result.num_elements
        total_transform_time += result.timing.transformation_time_ms
        total_dequant_time += result.timing.dequantization_time_ms
        total_validation_time += result.timing.validation_time_ms
    
    # Aggregate storage metrics
    total_scales_bytes = sum(r.storage.scales_bytes for r in tensor_results)
    total_metadata_bytes = sum(r.storage.metadata_bytes for r in tensor_results)
    total_padding_bytes = sum(r.storage.padding_bytes for r in tensor_results)
    total_headers_bytes = sum(r.storage.tensor_headers_bytes for r in tensor_results)
    total_overhead_bytes = sum(r.storage.serialization_overhead_bytes for r in tensor_results)
    
    theoretical_total = sum(r.storage.theoretical_bytes for r in tensor_results)
    actual_total = sum(r.storage.actual_bytes for r in tensor_results)
    
    aggregate_storage = StorageMetrics(
        theoretical_bytes=theoretical_total,
        actual_bytes=actual_total,
        metadata_bytes=total_metadata_bytes,
        scales_bytes=total_scales_bytes,
        padding_bytes=total_padding_bytes,
        tensor_headers_bytes=total_headers_bytes,
        serialization_overhead_bytes=total_overhead_bytes,
        overhead_ratio=actual_total / theoretical_total if theoretical_total > 0 else 0.0,
    )
    
    total_timing = TimingMetrics(
        transformation_time_ms=total_transform_time,
        dequantization_time_ms=total_dequant_time,
        validation_time_ms=total_validation_time,
        total_time_ms=total_transform_time + total_dequant_time + total_validation_time,
    )
    
    aggregate_compression_ratio = (
        total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else float('inf')
    )
    aggregate_bits_per_weight = (total_compressed_bytes * 8) / total_elements if total_elements > 0 else 0.0
    
    return ModelBenchmarkResult(
        model_identifier=model_identifier,
        parameter_count=total_elements,
        tensor_count=len(tensors),
        tensor_results=tensor_results,
        aggregate_compression_ratio=aggregate_compression_ratio,
        aggregate_bits_per_weight=aggregate_bits_per_weight,
        aggregate_original_bytes=total_original_bytes,
        aggregate_compressed_bytes=total_compressed_bytes,
        aggregate_storage=aggregate_storage,
        total_timing=total_timing,
        random_seed=random_seed,
    )


def run_method_comparison(
    tensor: np.ndarray,
    tensor_name: str,
    methods: List[Tuple[str, int, int, np.dtype]],  # (method, target_bits, group_size, scale_dtype)
    random_seed: int = 42,
) -> List[TensorBenchmarkResult]:
    """
    Run benchmark comparison across multiple compression methods on the same tensor.
    
    Args:
        tensor: Input tensor
        tensor_name: Tensor identifier
        methods: List of (method, target_bits, group_size, scale_dtype) tuples
        random_seed: Random seed for reproducibility
        
    Returns:
        List of TensorBenchmarkResult for each method
    """
    results = []
    for method, target_bits, group_size, scale_dtype in methods:
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name=tensor_name,
            compression_method=method,
            target_bits=target_bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            random_seed=random_seed,
        )
        results.append(result)
    return results


def run_repeatability_benchmark(
    tensor: np.ndarray,
    tensor_name: str,
    compression_method: str,
    target_bits: int,
    group_size: int,
    scale_dtype: np.dtype,
    num_runs: int = 10,
    base_seed: int = 42,
) -> Tuple[List[TensorBenchmarkResult], Dict[str, Any]]:
    """
    Run the same compression experiment multiple times to verify repeatability.
    
    For deterministic methods (all current LDMARK methods), results should be identical.
    Records random seeds for each run.
    
    Args:
        tensor: Input tensor
        tensor_name: Tensor identifier
        compression_method: Compression method
        target_bits: Target bits
        group_size: Group size
        scale_dtype: Scale dtype
        num_runs: Number of repeated runs
        base_seed: Base random seed
        
    Returns:
        Tuple of (list of results, repeatability analysis dict)
    """
    results = []
    for i in range(num_runs):
        seed = base_seed + i
        result = run_tensor_benchmark(
            tensor=tensor,
            tensor_name=tensor_name,
            compression_method=compression_method,
            target_bits=target_bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            random_seed=seed,
        )
        results.append(result)
    
    # Analyze repeatability
    if num_runs > 1:
        # Check if all dequantized results are identical
        first_dequant = results[0].error_distribution.mean  # Using mean as proxy
        all_identical = all(
            np.isclose(r.error_distribution.mean, first_dequant) 
            for r in results
        )
        
        # Check bits/weight consistency
        bpw_values = [r.actual_bits_per_weight for r in results]
        bpw_std = float(np.std(bpw_values))
        bpw_mean = float(np.mean(bpw_values))
        
        # Check timing variation
        timing_values = [r.timing.total_time_ms for r in results]
        timing_std = float(np.std(timing_values))
        timing_mean = float(np.mean(timing_values))
        
        analysis = {
            "num_runs": num_runs,
            "all_results_identical": all_identical,
            "bits_per_weight": {
                "mean": bpw_mean,
                "std": bpw_std,
                "min": float(np.min(bpw_values)),
                "max": float(np.max(bpw_values)),
            },
            "timing_ms": {
                "mean": timing_mean,
                "std": timing_std,
                "min": float(np.min(timing_values)),
                "max": float(np.max(timing_values)),
            },
            "seeds_used": [base_seed + i for i in range(num_runs)],
        }
    else:
        analysis = {
            "num_runs": 1,
            "all_results_identical": True,
            "bits_per_weight": {"mean": results[0].actual_bits_per_weight, "std": 0.0},
            "timing_ms": {"mean": results[0].timing.total_time_ms, "std": 0.0},
            "seeds_used": [base_seed],
        }
    
    return results, analysis


def create_synthetic_model(
    tensor_specs: List[Tuple[str, tuple, str]],
    generator: str = "random_normal",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Create a synthetic model with specified tensor shapes and dtypes.
    
    Args:
        tensor_specs: List of (name, shape, dtype) tuples
        generator: Tensor generator type
        seed: Random seed
        
    Returns:
        Dictionary of tensor name -> tensor array
    """
    tensors = {}
    rng = np.random.default_rng(seed)
    
    for i, (name, shape, dtype_str) in enumerate(tensor_specs):
        dtype = np.dtype(dtype_str)
        tensor_seed = seed + i
        tensors[name] = generate_test_tensor(
            shape=shape,
            dtype=dtype,
            generator=generator,
            seed=tensor_seed,
        )
    
    return tensors