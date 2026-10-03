"""
Error analysis for mixed-precision strategies.

Measures reconstruction error for individual tensors under different precisions.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from .plan import (
    PrecisionType, 
    TensorClassification, 
    TensorPrecisionAssignment,
    PrecisionCandidate,
    MixedPrecisionPlan,
)
from .storage import MixedPrecisionStorageModel


@dataclass(frozen=True)
class ErrorMeasurement:
    """Reconstruction error measurement for a tensor at a specific precision."""
    tensor_name: str
    precision: PrecisionType
    group_size: Optional[int]
    mae: float
    mse: float
    rmse: float
    max_absolute_error: float
    relative_error: float
    quantization_time_ms: float
    dequantization_time_ms: float
    effective_bits_per_weight: float
    storage_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "precision": self.precision.value,
            "group_size": self.group_size,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "max_absolute_error": self.max_absolute_error,
            "relative_error": self.relative_error,
            "quantization_time_ms": self.quantization_time_ms,
            "dequantization_time_ms": self.dequantization_time_ms,
            "effective_bits_per_weight": self.effective_bits_per_weight,
            "storage_bytes": self.storage_bytes,
        }


@dataclass(frozen=True)
class TensorErrorProfile:
    """Complete error profile for a tensor across all candidate precisions."""
    tensor_name: str
    tensor_classification: TensorClassification
    measurements: Dict[PrecisionType, ErrorMeasurement] = field(default_factory=dict)
    baseline_fp32_mae: float = 0.0
    baseline_fp32_mse: float = 0.0
    
    def get_measurement(self, precision: PrecisionType) -> Optional[ErrorMeasurement]:
        return self.measurements.get(precision)
    
    def get_error_delta(self, from_prec: PrecisionType, to_prec: PrecisionType) -> Optional[float]:
        """Get MAE increase from changing precision."""
        from_m = self.measurements.get(from_prec)
        to_m = self.measurements.get(to_prec)
        if from_m and to_m:
            return to_m.mae - from_m.mae
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "tensor_classification": self.tensor_classification.to_dict(),
            "measurements": {p.value: m.to_dict() for p, m in self.measurements.items()},
            "baseline_fp32_mae": self.baseline_fp32_mae,
            "baseline_fp32_mse": self.baseline_fp32_mse,
        }


def calculate_error_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Calculate error metrics between original and reconstructed tensors.
    
    Returns:
        (mae, mse, rmse, max_abs_error, relative_error)
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    abs_diff = np.abs(diff)
    
    mae = float(np.mean(abs_diff))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    max_abs_error = float(np.max(abs_diff))
    
    orig_norm = np.linalg.norm(original.astype(np.float64))
    if orig_norm > 0:
        relative_error = float(np.linalg.norm(diff) / orig_norm)
    else:
        relative_error = 0.0
    
    return mae, mse, rmse, max_abs_error, relative_error


def quantize_tensor(
    tensor: np.ndarray,
    precision: PrecisionType,
    group_size: int = 128,
) -> Tuple[np.ndarray, float, float]:
    """
    Quantize and dequantize a tensor.
    
    Returns:
        (reconstructed_tensor, quantization_time_ms, dequantization_time_ms)
    """
    # Import LDMARK compression utilities
    import sys
    sys.path.insert(0, "/Users/net/Desktop/Web Dev Projects/Eureka-2026/src")
    from ldmark.compression import (
        QuantizationConfig,
        QuantizationTarget,
        quantize_groupwise,
        dequantize_groupwise,
        BinaryConfig,
        TernaryConfig,
        binary_quantize_sign,
        binary_dequantize,
        ternary_quantize_threshold,
        ternary_dequantize,
    )
    
    if precision == PrecisionType.FP32:
        return tensor.astype(np.float32), 0.0, 0.0
    
    elif precision == PrecisionType.FP16:
        start = time.perf_counter()
        fp16 = tensor.astype(np.float16)
        q_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        reconstructed = fp16.astype(np.float32)
        dq_time = (time.perf_counter() - start) * 1000
        return reconstructed, q_time, dq_time
    
    elif precision == PrecisionType.INT8:
        config = QuantizationConfig(
            target_bits=QuantizationTarget.INT8,
            group_size=group_size,
            scale_dtype=np.float16,
            symmetric=True,
        )
        start = time.perf_counter()
        qtensor = quantize_groupwise(tensor, config)
        q_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        reconstructed = dequantize_groupwise(qtensor)
        dq_time = (time.perf_counter() - start) * 1000
        return reconstructed, q_time, dq_time
    
    elif precision == PrecisionType.INT4:
        config = QuantizationConfig(
            target_bits=QuantizationTarget.INT4,
            group_size=group_size,
            scale_dtype=np.float16,
            symmetric=True,
        )
        start = time.perf_counter()
        qtensor = quantize_groupwise(tensor, config)
        q_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        reconstructed = dequantize_groupwise(qtensor)
        dq_time = (time.perf_counter() - start) * 1000
        return reconstructed, q_time, dq_time
    
    elif precision == PrecisionType.BINARY:
        config = BinaryConfig(group_size=group_size, scale_dtype=np.float16)
        start = time.perf_counter()
        btensor = binary_quantize_sign(tensor, config)
        q_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        reconstructed = binary_dequantize(btensor)
        dq_time = (time.perf_counter() - start) * 1000
        return reconstructed, q_time, dq_time
    
    elif precision == PrecisionType.TERNARY:
        config = TernaryConfig(group_size=group_size, scale_dtype=np.float16)
        start = time.perf_counter()
        ttensor = ternary_quantize_threshold(tensor, config, threshold=0.05)
        q_time = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        reconstructed = ternary_dequantize(ttensor)
        dq_time = (time.perf_counter() - start) * 1000
        return reconstructed, q_time, dq_time
    
    else:
        raise ValueError(f"Unsupported precision: {precision}")


def measure_tensor_error(
    tensor: np.ndarray,
    tensor_name: str,
    classification: TensorClassification,
    precision: PrecisionType,
    group_size: int = 128,
    storage_model: Optional[MixedPrecisionStorageModel] = None,
) -> ErrorMeasurement:
    """Measure reconstruction error for a tensor at a specific precision."""
    reconstructed, q_time, dq_time = quantize_tensor(tensor, precision, group_size)
    
    mae, mse, rmse, max_err, rel_err = calculate_error_metrics(tensor, reconstructed)
    
    # Calculate storage
    if storage_model:
        candidate = PrecisionCandidate(
            precision=precision,
            group_size=group_size,
        )
        assignment = TensorPrecisionAssignment(
            tensor_name=tensor_name,
            precision=precision,
            group_size=group_size,
            candidate=candidate,
        )
        storage_comp = storage_model.calculate_tensor_storage(assignment, classification)
        storage_bytes = storage_comp.total_bytes
        effective_bpw = storage_comp.bits_per_weight
    else:
        storage_bytes = 0
        effective_bpw = precision.bits_per_weight
    
    return ErrorMeasurement(
        tensor_name=tensor_name,
        precision=precision,
        group_size=group_size,
        mae=mae,
        mse=mse,
        rmse=rmse,
        max_absolute_error=max_err,
        relative_error=rel_err,
        quantization_time_ms=q_time,
        dequantization_time_ms=dq_time,
        effective_bits_per_weight=effective_bpw,
        storage_bytes=storage_bytes,
    )


def measure_tensor_errors(
    tensor: np.ndarray,
    tensor_name: str,
    classification: TensorClassification,
    precisions: List[PrecisionType],
    group_size: int = 128,
    storage_model: Optional[MixedPrecisionStorageModel] = None,
) -> TensorErrorProfile:
    """Measure errors for a tensor across multiple precisions."""
    measurements = {}
    
    for prec in precisions:
        if prec == PrecisionType.FP32:
            # FP32 is lossless (up to float32 precision)
            measurements[prec] = ErrorMeasurement(
                tensor_name=tensor_name,
                precision=prec,
                group_size=None,
                mae=0.0,
                mse=0.0,
                rmse=0.0,
                max_absolute_error=0.0,
                relative_error=0.0,
                quantization_time_ms=0.0,
                dequantization_time_ms=0.0,
                effective_bits_per_weight=32.0,
                storage_bytes=tensor.nbytes,
            )
        else:
            measurements[prec] = measure_tensor_error(
                tensor, tensor_name, classification, prec, group_size, storage_model
            )
    
    # Baseline FP32 error (zero)
    baseline_mae, baseline_mse, _, _, _ = calculate_error_metrics(tensor, tensor)
    
    return TensorErrorProfile(
        tensor_name=tensor_name,
        tensor_classification=classification,
        measurements=measurements,
        baseline_fp32_mae=baseline_mae,
        baseline_fp32_mse=baseline_mse,
    )


def run_sensitivity_experiment(
    tensors: Dict[str, np.ndarray],
    classifications: Dict[str, TensorClassification],
    precisions: Optional[List[PrecisionType]] = None,
    group_size: int = 128,
    storage_model: Optional[MixedPrecisionStorageModel] = None,
) -> Dict[str, TensorErrorProfile]:
    """
    Run sensitivity experiment: compress one tensor at a time.
    
    This measures individual tensor reconstruction error, NOT end-to-end model quality.
    The results are useful for planning but do not directly predict model performance.
    
    Args:
        tensors: Dictionary of tensor name -> tensor array
        classifications: Dictionary of tensor name -> TensorClassification
        precisions: List of precisions to test (default: all quantized)
        group_size: Group size for grouped quantization
        storage_model: Storage model for accounting
        
    Returns:
        Dictionary of tensor name -> TensorErrorProfile
    """
    if precisions is None:
        precisions = [
            PrecisionType.FP16,
            PrecisionType.INT8,
            PrecisionType.INT4,
            PrecisionType.BINARY,
            PrecisionType.TERNARY,
        ]
    
    results = {}
    for name, tensor in tensors.items():
        if name not in classifications:
            continue
        classification = classifications[name]
        print(f"  Measuring {name} ({classification.role.value})...")
        profile = measure_tensor_errors(
            tensor, name, classification, precisions, group_size, storage_model
        )
        results[name] = profile
    
    return results


def create_error_profile_from_measurements(
    measurements: Dict[str, Dict[PrecisionType, ErrorMeasurement]],
    classifications: Dict[str, TensorClassification],
) -> Dict[str, TensorErrorProfile]:
    """Create error profiles from pre-computed measurements."""
    profiles = {}
    for name, meas in measurements.items():
        if name not in classifications:
            continue
        profiles[name] = TensorErrorProfile(
            tensor_name=name,
            tensor_classification=classifications[name],
            measurements=meas,
        )
    return profiles