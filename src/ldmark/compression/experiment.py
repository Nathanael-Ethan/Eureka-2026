"""
Experiment framework for LDMARK compression laboratory.

Provides reproducible experiment execution with structured metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import uuid

from .quantize import (
    QuantizationConfig,
    QuantizedTensor,
    quantize_groupwise,
    dequantize_groupwise,
    QuantizationTarget,
)
from .metrics import (
    CompressionMetrics,
    compute_full_metrics,
    calculate_error_metrics,
)
from .binary_ternary import (
    BinaryConfig,
    TernaryConfig,
    binary_quantize_sign,
    binary_dequantize,
    ternary_quantize_threshold,
    ternary_dequantize,
    calculate_binary_storage,
    calculate_ternary_storage,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a compression experiment."""
    experiment_name: str
    input_shape: tuple
    input_dtype: np.dtype
    target_bits: QuantizationTarget
    group_size: int
    scale_dtype: np.dtype = np.float16
    symmetric: bool = True
    seed: int = 42
    tensor_generator: str = "random_normal"  # "random_normal", "random_uniform", "ones", "custom"
    custom_tensor: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.tensor_generator == "custom" and self.custom_tensor is None:
            raise ValueError("custom_tensor required when tensor_generator='custom'")
        if self.custom_tensor is not None and self.custom_tensor.shape != self.input_shape:
            raise ValueError("custom_tensor shape must match input_shape")


@dataclass(frozen=True)
class ExperimentResult:
    """Result of a compression experiment."""
    experiment_id: str
    timestamp: str
    config: ExperimentConfig
    metrics: CompressionMetrics
    dequantized_tensor: np.ndarray
    quantized_tensor: QuantizedTensor
    error_metrics: Dict[str, float]
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": {
                "experiment_name": self.config.experiment_name,
                "input_shape": self.config.input_shape,
                "input_dtype": str(self.config.input_dtype),
                "target_bits": self.config.target_bits.name,
                "group_size": self.config.group_size,
                "scale_dtype": str(self.config.scale_dtype),
                "symmetric": self.config.symmetric,
                "seed": self.config.seed,
                "tensor_generator": self.config.tensor_generator,
            },
            "metrics": self.metrics.to_dict(),
            "error_metrics": self.error_metrics,
            "notes": self.notes,
        }


def generate_test_tensor(
    shape: tuple,
    dtype: np.dtype,
    generator: str = "random_normal",
    seed: int = 42,
    custom: Optional[np.ndarray] = None
) -> np.ndarray:
    """Generate a test tensor for experiments."""
    if generator == "custom" and custom is not None:
        return custom.astype(dtype)
    
    rng = np.random.default_rng(seed)
    
    if generator == "random_normal":
        tensor = rng.normal(0, 1, size=shape).astype(dtype)
    elif generator == "random_uniform":
        tensor = rng.uniform(-1, 1, size=shape).astype(dtype)
    elif generator == "ones":
        tensor = np.ones(shape, dtype=dtype)
    elif generator == "zeros":
        tensor = np.zeros(shape, dtype=dtype)
    else:
        raise ValueError(f"Unknown generator: {generator}")
    
    return tensor


def run_quantization_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a complete quantization experiment.
    
    This function:
    1. Generates or uses provided input tensor
    2. Quantizes using group-wise quantization
    3. Dequantizes back to float
    4. Computes all metrics
    5. Returns structured result
    """
    # Generate input tensor
    tensor = generate_test_tensor(
        config.input_shape,
        config.input_dtype,
        config.tensor_generator,
        config.seed,
        config.custom_tensor
    )
    
    # Create quantization config
    qconfig = QuantizationConfig(
        target_bits=config.target_bits,
        group_size=config.group_size,
        scale_dtype=config.scale_dtype,
        symmetric=config.symmetric,
    )
    
    # Quantize
    qtensor = quantize_groupwise(tensor, qconfig)
    
    # Dequantize
    dequantized = dequantize_groupwise(qtensor)
    
    # Compute metrics
    metrics = compute_full_metrics(
        tensor,
        dequantized,
        qtensor.data,
        qtensor.scales,
        config.target_bits.value,
        config.group_size,
        config.scale_dtype,
    )
    
    # Additional error metrics
    mae, mse, max_err, rel_err = calculate_error_metrics(tensor, dequantized)
    error_metrics = {
        "mae": mae,
        "mse": mse,
        "max_absolute_error": max_err,
        "relative_error": rel_err,
    }
    
    return ExperimentResult(
        experiment_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        config=config,
        metrics=metrics,
        dequantized_tensor=dequantized,
        quantized_tensor=qtensor,
        error_metrics=error_metrics,
    )


def run_binary_experiment(
    config: ExperimentConfig,
    binary_config: BinaryConfig,
    threshold: float = 0.0
) -> ExperimentResult:
    """Run binary quantization experiment (research only)."""
    tensor = generate_test_tensor(
        config.input_shape,
        config.input_dtype,
        config.tensor_generator,
        config.seed,
        config.custom_tensor
    )
    
    btensor = binary_quantize_sign(tensor, binary_config)
    dequantized = binary_dequantize(btensor)
    
    # Compute metrics
    original_size = tensor.nbytes
    compressed_size = calculate_binary_storage(tensor.size, binary_config)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    bits_per_weight = 1.0 + binary_config.scale_overhead_bits_per_weight()
    
    mae, mse, max_err, rel_err = calculate_error_metrics(tensor, dequantized)
    
    metrics = CompressionMetrics(
        original_dtype=str(tensor.dtype),
        compressed_representation=f"BINARY_g{binary_config.group_size}",
        bits_per_weight=bits_per_weight,
        scale_overhead_bits=binary_config.scale_overhead_bits_per_weight(),
        mean_absolute_error=mae,
        mean_squared_error=mse,
        max_absolute_error=max_err,
        relative_error=rel_err,
        compression_ratio=compression_ratio,
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        estimated_storage_bytes=compressed_size,
        actual_storage_bytes=btensor.packed_data.nbytes + btensor.scales.nbytes,
    )
    
    return ExperimentResult(
        experiment_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        config=config,
        metrics=metrics,
        dequantized_tensor=dequantized,
        quantized_tensor=btensor,
        error_metrics={"mae": mae, "mse": mse, "max_absolute_error": max_err, "relative_error": rel_err},
        notes="Binary quantization (research only)",
    )


def run_ternary_experiment(
    config: ExperimentConfig,
    ternary_config: TernaryConfig,
    threshold: float = 0.05
) -> ExperimentResult:
    """Run ternary quantization experiment (research only)."""
    tensor = generate_test_tensor(
        config.input_shape,
        config.input_dtype,
        config.tensor_generator,
        config.seed,
        config.custom_tensor
    )
    
    ttensor = ternary_quantize_threshold(tensor, ternary_config, threshold)
    dequantized = ternary_dequantize(ttensor)
    
    # Compute metrics
    original_size = tensor.nbytes
    compressed_size = calculate_ternary_storage(tensor.size, ternary_config)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    bits_per_weight = ternary_config.bits_per_weight() + ternary_config.scale_overhead_bits_per_weight()
    
    mae, mse, max_err, rel_err = calculate_error_metrics(tensor, dequantized)
    
    metrics = CompressionMetrics(
        original_dtype=str(tensor.dtype),
        compressed_representation=f"TERNARY_g{ternary_config.group_size}",
        bits_per_weight=bits_per_weight,
        scale_overhead_bits=ternary_config.scale_overhead_bits_per_weight(),
        mean_absolute_error=mae,
        mean_squared_error=mse,
        max_absolute_error=max_err,
        relative_error=rel_err,
        compression_ratio=compression_ratio,
        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,
        estimated_storage_bytes=compressed_size,
        actual_storage_bytes=ttensor.packed_data.nbytes + ttensor.scales.nbytes,
    )
    
    return ExperimentResult(
        experiment_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        config=config,
        metrics=metrics,
        dequantized_tensor=dequantized,
        quantized_tensor=ttensor,
        error_metrics={"mae": mae, "mse": mse, "max_absolute_error": max_err, "relative_error": rel_err},
        notes="Ternary quantization (research only)",
    )


def run_experiment_suite(
    configs: List[ExperimentConfig],
    include_binary: bool = False,
    include_ternary: bool = False,
) -> List[ExperimentResult]:
    """Run multiple experiments in sequence."""
    results = []
    for config in configs:
        result = run_quantization_experiment(config)
        results.append(result)
        
        if include_binary:
            bconfig = BinaryConfig(group_size=config.group_size, scale_dtype=config.scale_dtype)
            bresult = run_binary_experiment(config, bconfig)
            results.append(bresult)
        
        if include_ternary:
            tconfig = TernaryConfig(group_size=config.group_size, scale_dtype=config.scale_dtype)
            tresult = run_ternary_experiment(config, tconfig)
            results.append(tresult)
    
    return results