from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .models import DType, TensorInfo, TensorTrainableStatus


_numpy_bfloat16 = getattr(np, "bfloat16", None)

NUMPY_TO_DTYPE = {
    np.float32: DType.FP32,
    np.float16: DType.FP16,
    np.int8: DType.INT8,
    np.uint8: DType.INT8,
}
if _numpy_bfloat16 is not None:
    NUMPY_TO_DTYPE[_numpy_bfloat16] = DType.BF16

DTYPE_TO_NUMPY = {
    DType.FP32: np.float32,
    DType.FP16: np.float16,
    DType.INT8: np.int8,
    DType.INT4: np.int8,
    DType.FP8: np.float16,
    DType.FP4: np.float16,
    DType.BIT1: np.uint8,
}
if _numpy_bfloat16 is not None:
    DTYPE_TO_NUMPY[DType.BF16] = _numpy_bfloat16

TORCH_TO_DTYPE = {
    "float32": DType.FP32,
    "float16": DType.FP16,
    "bfloat16": DType.BF16,
    "int8": DType.INT8,
    "int4": DType.INT4,
    "uint8": DType.INT8,
    "float8_e4m3fn": DType.FP8,
    "float8_e5m2": DType.FP8,
}


def infer_dtype_from_name(name: str, dtype_str: str) -> DType:
    dtype_lower = dtype_str.lower()
    if dtype_lower in TORCH_TO_DTYPE:
        return TORCH_TO_DTYPE[dtype_lower]
    if dtype_lower in NUMPY_TO_DTYPE:
        return NUMPY_TO_DTYPE[dtype_lower]
    if "fp32" in dtype_lower or "float32" in dtype_lower:
        return DType.FP32
    if "fp16" in dtype_lower or "float16" in dtype_lower or "half" in dtype_lower:
        return DType.FP16
    if "bf16" in dtype_lower or "bfloat16" in dtype_lower:
        return DType.BF16
    if "int8" in dtype_lower:
        return DType.INT8
    if "int4" in dtype_lower:
        return DType.INT4
    if "fp8" in dtype_lower or "float8" in dtype_lower:
        return DType.FP8
    if "fp4" in dtype_lower or "float4" in dtype_lower:
        return DType.FP4
    if "1bit" in dtype_lower or "bit1" in dtype_lower or "bool" in dtype_lower:
        return DType.BIT1
    return DType.FP16


def create_tensor_info(
    name: str,
    shape: Union[List[int], Tuple[int, ...]],
    dtype: Union[DType, str, np.dtype, type],
    trainable: TensorTrainableStatus = TensorTrainableStatus.UNKNOWN,
) -> TensorInfo:
    if isinstance(dtype, str):
        dtype = infer_dtype_from_name(name, dtype)
    elif isinstance(dtype, np.dtype):
        dtype = NUMPY_TO_DTYPE.get(dtype, DType.FP16)
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype = NUMPY_TO_DTYPE.get(np.dtype(dtype), DType.FP16)
    return TensorInfo.from_shape_and_dtype(name, list(shape), dtype, trainable)


def analyze_numpy_dict(tensors: Dict[str, np.ndarray], trainable_names: Optional[set] = None) -> List[TensorInfo]:
    trainable_names = trainable_names or set()
    results = []
    for name, array in tensors.items():
        dtype = NUMPY_TO_DTYPE.get(array.dtype.type, DType.FP16)
        trainable = TensorTrainableStatus.TRAINABLE if name in trainable_names else TensorTrainableStatus.FROZEN
        results.append(create_tensor_info(name, array.shape, dtype, trainable))
    return results


def analyze_torch_state_dict(state_dict: Dict[str, Any], trainable_names: Optional[set] = None) -> List[TensorInfo]:
    trainable_names = trainable_names or set()
    results = []
    for name, tensor in state_dict.items():
        if hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
            shape = list(tensor.shape)
            dtype_str = str(tensor.dtype).split(".")[-1]
            dtype = infer_dtype_from_name(name, dtype_str)
            trainable = TensorTrainableStatus.TRAINABLE if name in trainable_names else TensorTrainableStatus.FROZEN
            results.append(create_tensor_info(name, shape, dtype, trainable))
    return results


def calculate_tensor_statistics(tensors: List[TensorInfo]) -> Dict[str, Any]:
    if not tensors:
        return {
            "total_tensors": 0,
            "total_elements": 0,
            "total_raw_bytes": 0,
            "by_dtype": {},
            "shape_stats": {},
        }

    total_elements = sum(t.num_elements for t in tensors)
    total_raw_bytes = sum(t.estimated_raw_storage_bytes for t in tensors)

    by_dtype: Dict[str, Dict[str, int]] = {}
    for t in tensors:
        key = t.dtype.value
        if key not in by_dtype:
            by_dtype[key] = {"count": 0, "elements": 0, "bytes": 0}
        by_dtype[key]["count"] += 1
        by_dtype[key]["elements"] += t.num_elements
        by_dtype[key]["bytes"] += t.estimated_raw_storage_bytes

    shapes = [tuple(t.shape) for t in tensors]
    unique_shapes = set(shapes)
    shape_stats = {
        "unique_shapes": len(unique_shapes),
        "max_dims": max(len(s) for s in shapes) if shapes else 0,
        "min_dims": min(len(s) for s in shapes) if shapes else 0,
    }

    return {
        "total_tensors": len(tensors),
        "total_elements": total_elements,
        "total_raw_bytes": total_raw_bytes,
        "by_dtype": by_dtype,
        "shape_stats": shape_stats,
    }