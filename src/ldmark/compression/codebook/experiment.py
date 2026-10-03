"""
Codebook experiment framework for LDMARK.

Provides reproducible experiment execution comparing uniform vs codebook quantization.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

from .representation import (
    CodebookConfig,
    CodebookTensor,
    create_codebook_tensor,
    reconstruct_from_codebook,
)
from .generation import generate_codebook_kmeans, generate_codebook_groupwise
from .analysis import analyze_weight_distribution, WeightDistributionStats
from .storage import (
    calculate_codebook_storage,
    calculate_uniform_storage_comparison,
    StorageAccounting,
)
from ..quantize import quantize_int4, QuantizationConfig, QuantizationTarget
from ..metrics import calculate_error_metrics, calculate_compression_ratio


@dataclass(frozen=True)
class CodebookExperimentConfig:
    """Configuration for a codebook quantization experiment."""
    experiment_name: str
    tensor_shape: Tuple[int, ...]
    source_dtype: np.dtype
    codebook_size: int
    group_size: Optional[int] = None
    max_iterations: int = 20
    tolerance: float = 1e-4
    seed: int = 42
    initialization: str = "percentile"
    tensor_generator: str = "random_normal"  # "random_normal", "random_uniform", "ones", "zeros", "heavy_tail"
    custom_tensor: Optional[np.ndarray] = None
    compare_uniform_int4: bool = True
    
    def __post_init__(self):
        if self.tensor_generator == "custom" and self.custom_tensor is None:
            raise ValueError("custom_tensor required when tensor_generator='custom'")
        if self.custom_tensor is not None and self.custom_tensor.shape != self.tensor_shape:
            raise ValueError("custom_tensor shape must match tensor_shape")
    
    def to_codebook_config(self) -> CodebookConfig:
        """Convert to CodebookConfig."""
        return CodebookConfig(
            codebook_size=self.codebook_size,
            group_size=self.group_size,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            seed=self.seed,
            initialization=self.initialization,
        )


@dataclass(frozen=True)
class CodebookExperimentResult:
    """Result of a codebook quantization experiment."""
    experiment_id: str
    timestamp: str
    config: CodebookExperimentConfig
    weight_distribution: WeightDistributionStats
    codebook_storage: StorageAccounting
    uniform_storage: Optional[Dict[str, Any]] = None
    codebook_error: Optional[Dict[str, float]] = None
    uniform_error: Optional[Dict[str, float]] = None
    generation_time_ms: float = 0.0
    quantization_time_ms: float = 0.0
    dequantization_time_ms: float = 0.0
    total_time_ms: float = 0.0
    error_history: List[float] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": {
                "experiment_name": self.config.experiment_name,
                "tensor_shape": self.config.tensor_shape,
                "source_dtype": str(self.config.source_dtype),
                "codebook_size": self.config.codebook_size,
                "group_size": self.config.group_size,
                "max_iterations": self.config.max_iterations,
                "tolerance": self.config.tolerance,
                "seed": self.config.seed,
                "initialization": self.config.initialization,
                "tensor_generator": self.config.tensor_generator,
            },
            "weight_distribution": self.weight_distribution.to_dict(),
            "codebook_storage": self.codebook_storage.to_dict(),
            "generation_time_ms": self.generation_time_ms,
            "quantization_time_ms": self.quantization_time_ms,
            "dequantization_time_ms": self.dequantization_time_ms,
            "total_time_ms": self.total_time_ms,
            "error_history": self.error_history,
            "notes": self.notes,
        }
        
        if self.uniform_storage is not None:
            d["uniform_storage"] = self.uniform_storage
        if self.codebook_error is not None:
            d["codebook_error"] = self.codebook_error
        if self.uniform_error is not None:
            d["uniform_error"] = self.uniform_error
        
        return d
    
    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


def generate_test_tensor(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    generator: str = "random_normal",
    seed: int = 42,
    custom: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate a test tensor for experiments."""
    if generator == "custom" and custom is not None:
        return custom.astype(dtype)
    
    rng = np.random.default_rng(seed)
    
    if generator == "random_normal":
        tensor = rng.normal(0, 1, size=shape).astype(dtype)
    elif generator == "random_uniform":
        tensor = rng.uniform(-1, 1, size=shape).astype(dtype)
    elif generator == "heavy_tail":
        # Student's t-distribution with df=3 for heavy tails
        tensor = rng.standard_t(df=3, size=shape).astype(dtype)
    elif generator == "ones":
        tensor = np.ones(shape, dtype=dtype)
    elif generator == "zeros":
        tensor = np.zeros(shape, dtype=dtype)
    elif generator == "transformer_like":
        # Simulate transformer weight distribution: mostly small, some large
        base = rng.normal(0, 0.02, size=shape)
        # Add some outlier weights
        outlier_mask = rng.random(shape) < 0.01
        base[outlier_mask] *= 10
        tensor = base.astype(dtype)
    else:
        raise ValueError(f"Unknown generator: {generator}")
    
    return tensor


def run_codebook_experiment(config: CodebookExperimentConfig) -> CodebookExperimentResult:
    """
    Run a complete codebook quantization experiment.
    
    Steps:
    1. Generate or load test tensor
    2. Analyze weight distribution
    3. Generate codebook using k-means
    4. Quantize tensor using codebook
    5. Reconstruct and measure errors
    6. Calculate storage
    7. Optionally compare with uniform INT4
    
    Returns:
        CodebookExperimentResult with all metrics
    """
    start_total = time.perf_counter()
    
    # Generate tensor
    tensor = generate_test_tensor(
        config.tensor_shape,
        config.source_dtype,
        config.tensor_generator,
        config.seed,
        config.custom_tensor,
    )
    
    # Analyze weight distribution
    weight_dist = analyze_weight_distribution(tensor)
    
    # Create codebook config
    cb_config = config.to_codebook_config()
    
    # Generate codebook
    gen_start = time.perf_counter()
    if cb_config.is_group_wise:
        group_results = generate_codebook_groupwise(tensor, cb_config)
        # Flatten results
        all_indices = np.concatenate([r[1] for r in group_results])
        codebooks = [r[0] for r in group_results]
        error_history = group_results[0][2] if group_results else []
    else:
        codebook, indices, error_history = generate_codebook_kmeans(tensor, cb_config)
        codebooks = [codebook]
        all_indices = indices
    generation_time = (time.perf_counter() - gen_start) * 1000
    
    # Create CodebookTensor
    cb_tensor = create_codebook_tensor(tensor, cb_config, codebooks, 
                                       [all_indices] if not cb_config.is_group_wise else [r[1] for r in group_results])
    
    # Reconstruct
    dequant_start = time.perf_counter()
    reconstructed = reconstruct_from_codebook(cb_tensor)
    dequant_time = (time.perf_counter() - dequant_start) * 1000
    
    # Compute errors
    mae, mse, max_err, rel_err = calculate_error_metrics(tensor, reconstructed)
    codebook_error = {
        "mae": mae,
        "mse": mse,
        "rmse": np.sqrt(mse),
        "max_absolute_error": max_err,
        "relative_error": rel_err,
    }
    
    # Storage accounting
    cb_storage = calculate_codebook_storage(cb_tensor)
    
    # Uniform INT4 comparison
    uniform_storage = None
    uniform_error = None
    uniform_time = 0.0
    
    if config.compare_uniform_int4 and tensor.dtype == np.float32:
        uniform_start = time.perf_counter()
        qconfig = QuantizationConfig(
            target_bits=QuantizationTarget.INT4,
            group_size=config.group_size if config.group_size else 128,
            scale_dtype=np.float16,
            symmetric=True,
        )
        qtensor = quantize_int4(tensor, group_size=qconfig.group_size)
        uniform_time = (time.perf_counter() - uniform_start) * 1000
        
        # Dequantize uniform
        from ..quantize import dequantize_int4
        uniform_dequant = dequantize_int4(qtensor)
        
        mae_u, mse_u, max_err_u, rel_err_u = calculate_error_metrics(tensor, uniform_dequant)
        uniform_error = {
            "mae": mae_u,
            "mse": mse_u,
            "rmse": np.sqrt(mse_u),
            "max_absolute_error": max_err_u,
            "relative_error": rel_err_u,
        }
        
        # Uniform storage
        uniform_storage = calculate_uniform_storage_comparison(
            tensor.nbytes,
            4,
            config.group_size if config.group_size else 128,
            16,
        )
    
    total_time = (time.perf_counter() - start_total) * 1000
    
    return CodebookExperimentResult(
        experiment_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now().isoformat(),
        config=config,
        weight_distribution=weight_dist,
        codebook_storage=cb_storage,
        uniform_storage=uniform_storage,
        codebook_error=codebook_error,
        uniform_error=uniform_error,
        generation_time_ms=generation_time,
        quantization_time_ms=uniform_time if config.compare_uniform_int4 else 0,
        dequantization_time_ms=dequant_time,
        total_time_ms=total_time,
        error_history=error_history,
    )


def run_uniform_vs_codebook_experiment(
    tensor: np.ndarray,
    codebook_sizes: List[int] = None,
    group_sizes: List[int] = None,
    seed: int = 42,
) -> List[CodebookExperimentResult]:
    """
    Run comprehensive comparison of uniform INT4 vs codebook quantization.
    
    Args:
        tensor: Input tensor to quantize
        codebook_sizes: List of codebook sizes to test
        group_sizes: List of group sizes (None for global)
        seed: Random seed
        
    Returns:
        List of experiment results
    """
    if codebook_sizes is None:
        codebook_sizes = [2, 4, 8, 16, 32]
    if group_sizes is None:
        group_sizes = [None, 128, 64]
    
    results = []
    for group_size in group_sizes:
        for cb_size in codebook_sizes:
            if group_size is None:
                # Global codebook
                config = CodebookExperimentConfig(
                    experiment_name=f"codebook_global_k{cb_size}",
                    tensor_shape=tensor.shape,
                    source_dtype=tensor.dtype,
                    codebook_size=cb_size,
                    group_size=None,
                    seed=seed,
                    tensor_generator="custom",
                    custom_tensor=tensor,
                )
            else:
                # Group-wise codebook
                config = CodebookExperimentConfig(
                    experiment_name=f"codebook_group{group_size}_k{cb_size}",
                    tensor_shape=tensor.shape,
                    source_dtype=tensor.dtype,
                    codebook_size=cb_size,
                    group_size=group_size,
                    seed=seed,
                    tensor_generator="custom",
                    custom_tensor=tensor,
                )
            
            result = run_codebook_experiment(config)
            results.append(result)
    
    return results


def run_binary_codebook_experiment(
    tensor: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare fixed sign binary vs 2-value learned codebook.
    
    This is the 1-bit research connection experiment.
    
    Args:
        tensor: Input tensor
        seed: Random seed
        
    Returns:
        Comparison results dictionary
    """
    from ..binary_ternary import (
        BinaryConfig,
        binary_quantize_sign,
        binary_dequantize,
        calculate_binary_storage,
    )
    
    # Fixed sign binary
    bconfig = BinaryConfig(group_size=128, scale_dtype=np.float16)
    btensor = binary_quantize_sign(tensor, bconfig)
    binary_reconstructed = binary_dequantize(btensor)
    
    binary_mae, binary_mse, binary_max, binary_rel = calculate_error_metrics(
        tensor, binary_reconstructed
    )
    binary_storage = calculate_binary_storage(tensor.size, bconfig)
    
    # 2-value learned codebook
    cb_config = CodebookConfig(
        codebook_size=2,
        group_size=128,
        max_iterations=50,
        tolerance=1e-6,
        seed=seed,
        initialization="percentile",
    )
    
    cb_results = generate_codebook_groupwise(tensor, cb_config)
    codebooks = [r[0] for r in cb_results]
    indices_list = [r[1] for r in cb_results]
    
    cb_tensor = create_codebook_tensor(tensor, cb_config, codebooks, indices_list)
    cb_reconstructed = reconstruct_from_codebook(cb_tensor)
    
    cb_mae, cb_mse, cb_max, cb_rel = calculate_error_metrics(
        tensor, cb_reconstructed
    )
    cb_storage = calculate_codebook_storage(cb_tensor)
    
    return {
        "tensor_shape": tensor.shape,
        "fixed_binary": {
            "mae": binary_mae,
            "mse": binary_mse,
            "max_abs_error": binary_max,
            "relative_error": binary_rel,
            "storage_bytes": binary_storage,
            "bpw": (binary_storage * 8) / tensor.size,
        },
        "learned_binary": {
            "mae": cb_mae,
            "mse": cb_mse,
            "max_abs_error": cb_max,
            "relative_error": cb_rel,
            "storage_bytes": cb_storage.total_bytes,
            "bpw": cb_storage.effective_bits_per_weight,
            "codebooks": [cb.tolist() for cb in codebooks],
        },
        "improvement": {
            "mae_reduction": (binary_mae - cb_mae) / binary_mae if binary_mae > 0 else 0,
            "mse_reduction": (binary_mse - cb_mse) / binary_mse if binary_mse > 0 else 0,
            "max_error_reduction": (binary_max - cb_max) / binary_max if binary_max > 0 else 0,
        },
    }