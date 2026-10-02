"""
PrismML Q1_0_g128 Research Benchmark for LDMARK.

Validates the mathematical representation of PrismML's Q1_0_g128 format:
- 128 one-bit weights (128 bits)
- 1 FP16 scale (16 bits)
- Total: 144 bits for 128 weights = 1.125 bits/weight

This is a RESEARCH CALCULATOR ONLY - NOT an implementation of PrismML.
Clearly distinguishes LDMARK mathematical representation experiment
from PrismML's published model results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import json

from .model import (
    BenchmarkRecord,
    TensorBenchmarkResult,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
    HardwareMetadata,
)
from .runner import run_tensor_benchmark
from .storage import calculate_theoretical_storage, validate_storage
from ..compression.prismml_calc import (
    prismml_q1_0_g128_bits_per_weight,
    calculate_prismml_storage,
    validate_prismml_claim,
    PrismMLConfig,
)


@dataclass(frozen=True)
class PrismMLBenchmarkResult:
    """Result of PrismML Q1_0_g128 mathematical validation benchmark."""
    # Mathematical validation
    theoretical_bpw: float  # Should be 1.125
    calculated_bpw: float   # From actual quantization
    bpw_error: float        # Absolute difference
    bpw_relative_error: float
    
    # Storage validation
    theoretical_storage_bytes: int
    actual_storage_bytes: int
    storage_overhead_ratio: float
    storage_validation_passed: bool
    
    # Error metrics (on random tensor)
    mae: float
    mse: float
    max_absolute_error: float
    relative_error: float
    error_distribution: ErrorDistribution
    
    # Timing
    quantization_time_ms: float
    dequantization_time_ms: float
    validation_time_ms: float
    
    # Distinction notice
    ldmark_representation_experiment: bool = True
    prismml_model_quality_claim: bool = False
    disclaimer: str = (
        "This benchmark validates the LDMARK mathematical representation "
        "of Q1_0_g128 (1-bit weights + FP16 scale per 128-group). "
        "It does NOT reproduce PrismML's model quality, training methodology, "
        "or published results. LDMARK representation experiment ≠ PrismML model."
    )
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "theoretical_bpw": self.theoretical_bpw,
            "calculated_bpw": self.calculated_bpw,
            "bpw_error": self.bpw_error,
            "bpw_relative_error": self.bpw_relative_error,
            "theoretical_storage_bytes": self.theoretical_storage_bytes,
            "actual_storage_bytes": self.actual_storage_bytes,
            "storage_overhead_ratio": self.storage_overhead_ratio,
            "storage_validation_passed": self.storage_validation_passed,
            "mae": self.mae,
            "mse": self.mse,
            "max_absolute_error": self.max_absolute_error,
            "relative_error": self.relative_error,
            "error_distribution": self.error_distribution.to_dict(),
            "quantization_time_ms": self.quantization_time_ms,
            "dequantization_time_ms": self.dequantization_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "ldmark_representation_experiment": self.ldmark_representation_experiment,
            "prismml_model_quality_claim": self.prismml_model_quality_claim,
            "disclaimer": self.disclaimer,
        }
        return d
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_prismml_q1_0_g128_benchmark(
    tensor: np.ndarray,
    tensor_name: str = "prismml_test_tensor",
    group_size: int = 128,
    scale_dtype: np.dtype = np.float16,
    random_seed: int = 42,
) -> PrismMLBenchmarkResult:
    """
    Run PrismML Q1_0_g128 mathematical validation benchmark.
    
    This validates the LDMARK mathematical representation of Q1_0_g128:
    - 1-bit weights per parameter (sign-only)
    - 1 FP16 scale per 128-weight group
    - Theoretical: 1.125 bits/weight
    
    IMPORTANT: This is a REPRESENTATION EXPERIMENT ONLY.
    It does NOT claim to reproduce PrismML's model quality or results.
    
    Args:
        tensor: Input tensor to quantize (FP32 or FP16)
        tensor_name: Name for the tensor
        group_size: Group size (default 128 for Q1_0_g128)
        scale_dtype: Scale data type (default FP16)
        random_seed: Random seed for reproducibility
        
    Returns:
        PrismMLBenchmarkResult with full validation metrics
    """
    # Theoretical calculation
    theoretical_bpw = prismml_q1_0_g128_bits_per_weight(group_size, 1, 16)
    
    # Use binary quantization as the closest LDMARK equivalent
    # Binary: 1-bit weights + FP16 scale per group
    from ..compression.binary_ternary import (
        BinaryConfig,
        binary_quantize_sign,
        binary_dequantize,
        calculate_binary_storage,
    )
    
    config = BinaryConfig(
        group_size=group_size,
        scale_dtype=scale_dtype,
    )
    
    # Quantize
    import time
    quant_start = time.perf_counter()
    btensor = binary_quantize_sign(tensor, config)
    quantization_time_ms = (time.perf_counter() - quant_start) * 1000
    
    # Dequantize
    dequant_start = time.perf_counter()
    dequantized = binary_dequantize(btensor)
    dequantization_time_ms = (time.perf_counter() - dequant_start) * 1000
    
    # Calculate error metrics
    from ..compression.metrics import calculate_error_metrics
    mae, mse, max_err, rel_err = calculate_error_metrics(tensor, dequantized)
    
    # Error distribution
    error_dist = ErrorDistribution.from_errors(
        np.abs(tensor.astype(np.float64) - dequantized.astype(np.float64)).flatten()
    )
    
    # Validation timing
    val_start = time.perf_counter()
    # (error metrics already computed)
    validation_time_ms = (time.perf_counter() - val_start) * 1000
    
    # Storage calculations
    num_weights = tensor.size
    theoretical_storage = calculate_theoretical_storage(
        num_weights, 1, group_size, 16
    )
    
    # Actual storage from binary representation
    actual_storage = btensor.packed_data.nbytes + btensor.scales.nbytes
    
    # Calculate actual bits per weight
    total_bits = actual_storage * 8
    calculated_bpw = total_bits / num_weights
    
    bpw_error = abs(calculated_bpw - theoretical_bpw)
    bpw_relative_error = bpw_error / theoretical_bpw if theoretical_bpw > 0 else 0.0
    
    # Storage validation
    storage_validation = validate_storage(
        num_weights=num_weights,
        bits_per_weight=1,
        group_size=group_size,
        quantized_data=btensor.packed_data,
        scales=btensor.scales,
        scale_bits=16,
    )
    
    return PrismMLBenchmarkResult(
        theoretical_bpw=theoretical_bpw,
        calculated_bpw=calculated_bpw,
        bpw_error=bpw_error,
        bpw_relative_error=bpw_relative_error,
        theoretical_storage_bytes=theoretical_storage,
        actual_storage_bytes=actual_storage,
        storage_overhead_ratio=storage_validation.overhead_ratio,
        storage_validation_passed=storage_validation.validation_passed,
        mae=mae,
        mse=mse,
        max_absolute_error=max_err,
        relative_error=rel_err,
        error_distribution=error_dist,
        quantization_time_ms=quantization_time_ms,
        dequantization_time_ms=dequantization_time_ms,
        validation_time_ms=validation_time_ms,
    )


def run_prismml_scaling_benchmark(
    model_params_list: List[int] = None,
) -> Dict[str, Any]:
    """
    Run PrismML storage scaling benchmark across model sizes.
    
    Validates the mathematical scaling formula:
    bits/weight = (group_size * weight_bits + scale_bits) / group_size
    
    For Q1_0_g128: (128 * 1 + 16) / 128 = 1.125 bits/weight
    
    Args:
        model_params_list: List of model parameter counts to test
        
    Returns:
        Dictionary with scaling validation results
    """
    if model_params_list is None:
        model_params_list = [
            1_000_000,        # 1M
            10_000_000,       # 10M
            100_000_000,      # 100M
            1_000_000_000,    # 1B
            7_000_000_000,    # 7B
            27_000_000_000,   # 27B (PrismML claim)
            70_000_000_000,   # 70B
        ]
    
    results = []
    for params in model_params_list:
        calc = calculate_prismml_storage(PrismMLConfig(model_params=params))
        results.append({
            "model_params": params,
            "bits_per_weight": calc["bits_per_weight"],
            "total_gb": calc["total_gb"],
            "group_count": calc["group_count"],
        })
    
    # Verify bpw is constant across all model sizes
    bpw_values = [r["bits_per_weight"] for r in results]
    bpw_constant = all(abs(b - 1.125) < 1e-9 for b in bpw_values)
    
    return {
        "bpw_constant_across_sizes": bpw_constant,
        "theoretical_bpw": 1.125,
        "results": results,
        "formula": "(group_size * weight_bits + scale_bits) / group_size",
        "formula_values": "128 * 1 + 16 = 144 bits per 128 weights = 1.125 bits/weight",
        "disclaimer": "This validates the mathematical storage formula only. "
                      "It does not validate PrismML's model quality claims.",
    }


def run_prismml_claim_validation(
    claimed_gb: float = 3.9,
    model_params: int = 27_000_000_000,
    tolerance: float = 0.1,
) -> Dict[str, Any]:
    """
    Validate PrismML's claimed storage for 27B model at Q1_0_g128.
    
    PrismML claims ~3.9 GB for 27B parameters at Q1_0_g128.
    LDMARK calculates: 27e9 * 1.125 / 8 / 1024^3 = ~3.54 GB
    
    This validation reports the mathematical difference.
    It does NOT judge which is correct - only reports the calculation.
    
    Args:
        claimed_gb: PrismML's claimed storage in GB
        model_params: Model parameter count
        tolerance: Relative error tolerance
        
    Returns:
        Validation result dictionary
    """
    validation = validate_prismml_claim(claimed_gb, model_params, tolerance)
    
    return {
        "claimed_gb": claimed_gb,
        "calculated_gb": validation["calculated_gb"],
        "difference_gb": validation["difference_gb"],
        "relative_error_percent": validation["relative_error"] * 100,
        "within_tolerance": validation["within_tolerance"],
        "tolerance_percent": tolerance * 100,
        "details": validation["details"],
        "disclaimer": (
            "This is a mathematical storage calculation validation. "
            "LDMARK calculates 3.54 GB for 27B params at 1.125 bits/weight. "
            "PrismML claims 3.9 GB. The difference may be due to "
            "metadata, alignment, tokenizer embeddings, or other factors "
            "not included in the pure weight storage calculation. "
            "This does not validate or invalidate PrismML's model quality."
        ),
    }


def create_prismml_comparison_table() -> List[Dict[str, Any]]:
    """
    Create comparison table of LDMARK representations vs PrismML Q1_0_g128.
    
    Returns factual measured values for:
    - FP32, FP16, INT8, INT4, Binary (Q1_0_g128 equivalent), Ternary
    
    No rankings or subjective quality labels.
    """
    representations = [
        ("FP32", 32, 0, 0, "No quantization"),
        ("FP16", 16, 0, 0, "No quantization"),
        ("INT8 (g128)", 8, 128, 16, "Group-wise INT8"),
        ("INT4 (g128)", 4, 128, 16, "Group-wise INT4"),
        ("Binary Q1_0_g128 (LDMARK)", 1, 128, 16, "1-bit + FP16 scale per 128-group"),
        ("Ternary (g128)", np.log2(3), 128, 16, "~1.585-bit + FP16 scale per 128-group"),
    ]
    
    model_params = 27_000_000_000
    
    results = []
    for name, w_bits, g_size, s_bits, desc in representations:
        if g_size == 0:
            bpw = w_bits
            total_bits = model_params * bpw
            group_count = 0
        else:
            bpw = (g_size * w_bits + s_bits) / g_size
            total_bits = model_params * bpw
            group_count = (model_params + g_size - 1) // g_size
        
        total_gb = total_bits / 8 / (1024 ** 3)
        
        results.append({
            "representation": name,
            "description": desc,
            "weight_bits": w_bits,
            "group_size": g_size,
            "scale_bits": s_bits,
            "bits_per_weight": bpw,
            "total_gb": total_gb,
            "group_count": group_count,
        })
    
    return results


def export_prismml_benchmark_report(
    result: PrismMLBenchmarkResult,
    filepath: str,
) -> None:
    """Export PrismML benchmark result to JSON."""
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)


def run_full_prismml_validation_suite(
    tensor_shapes: List[tuple] = None,
    output_dir: str = "prismml_benchmark_results",
) -> Dict[str, Any]:
    """
    Run complete PrismML Q1_0_g128 validation suite.
    
    Includes:
    1. Mathematical bpw validation
    2. Scaling validation across model sizes
    3. Claim validation (3.9 GB for 27B)
    4. Representation comparison table
    5. Tensor-level benchmark on test tensors
    
    Args:
        tensor_shapes: List of tensor shapes to test
        output_dir: Directory for output files
        
    Returns:
        Complete validation suite results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if tensor_shapes is None:
        tensor_shapes = [
            (128,),       # Exactly one group
            (256,),       # Two groups
            (1024,),      # Eight groups
            (512, 512),   # 2D tensor
            (4096, 1024), # Large 2D tensor
        ]
    
    # 1. Mathematical validation
    math_validation = {
        "formula": "(group_size * weight_bits + scale_bits) / group_size",
        "q1_0_g128": "(128 * 1 + 16) / 128 = 1.125",
        "bpw": prismml_q1_0_g128_bits_per_weight(),
    }
    
    # 2. Scaling validation
    scaling = run_prismml_scaling_benchmark()
    
    # 3. Claim validation
    claim_validation = run_prismml_claim_validation()
    
    # 4. Comparison table
    comparison = create_prismml_comparison_table()
    
    # 5. Tensor-level benchmarks
    tensor_results = []
    for shape in tensor_shapes:
        tensor = np.random.default_rng(42).normal(0, 1, shape).astype(np.float32)
        result = run_prismml_q1_0_g128_benchmark(
            tensor=tensor,
            tensor_name=f"shape_{shape}",
        )
        tensor_results.append({
            "shape": shape,
            "num_elements": tensor.size,
            "theoretical_bpw": result.theoretical_bpw,
            "calculated_bpw": result.calculated_bpw,
            "bpw_error": result.bpw_error,
            "mae": result.mae,
            "mse": result.mse,
            "max_error": result.max_absolute_error,
            "relative_error": result.relative_error,
            "storage_overhead_ratio": result.storage_overhead_ratio,
            "storage_validation_passed": result.storage_validation_passed,
        })
        
        # Export individual result
        export_prismml_benchmark_report(
            result,
            os.path.join(output_dir, f"prismml_q1_0_g128_{shape}.json"),
        )
    
    suite_result = {
        "mathematical_validation": math_validation,
        "scaling_validation": scaling,
        "claim_validation": claim_validation,
        "representation_comparison": comparison,
        "tensor_benchmarks": tensor_results,
        "disclaimer": (
            "ALL RESULTS ARE LDMARK MATHEMATICAL REPRESENTATION EXPERIMENTS. "
            "These benchmarks validate the storage mathematics of Q1_0_g128 "
            "(1-bit weights + FP16 scale per 128-group = 1.125 bits/weight). "
            "They do NOT reproduce PrismML's model architecture, training, "
            "or published quality results. LDMARK representation experiment "
            "≠ PrismML model."
        ),
    }
    
    # Export full suite
    with open(os.path.join(output_dir, "prismml_full_validation.json"), 'w') as f:
        json.dump(suite_result, f, indent=2)
    
    return suite_result