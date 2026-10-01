from __future__ import annotations

from typing import Dict, List, Optional, Union
from .models import DType, StorageEstimate, ParameterCounts


DEFAULT_DTYPES = [
    DType.FP32,
    DType.FP16,
    DType.BF16,
    DType.INT8,
    DType.INT4,
    DType.FP8,
    DType.FP4,
    DType.BIT1,
]


def estimate_storage_for_dtype(param_count: int, dtype: DType, description: str = "") -> StorageEstimate:
    return StorageEstimate.from_parameter_count(param_count, dtype, description)


def estimate_all_storages(param_count: int, dtypes: Optional[List[DType]] = None) -> List[StorageEstimate]:
    dtypes = dtypes or DEFAULT_DTYPES
    return [estimate_storage_for_dtype(param_count, dt) for dt in dtypes]


def estimate_storage_from_parameter_counts(counts: ParameterCounts, dtypes: Optional[List[DType]] = None) -> List[StorageEstimate]:
    dtypes = dtypes or DEFAULT_DTYPES
    results = []
    for dt in dtypes:
        dtype_count = counts.by_dtype.get(dt, 0)
        if dtype_count > 0:
            results.append(estimate_storage_for_dtype(dtype_count, dt, f"Theoretical {dt.value} storage for {dt.value} parameters"))
        else:
            total = counts.total
            results.append(estimate_storage_for_dtype(total, dt, f"Theoretical {dt.value} storage (assuming all parameters converted to {dt.value})"))
    return results


def compare_storage_estimates(estimates: List[StorageEstimate]) -> Dict[str, Any]:
    if not estimates:
        return {}

    baseline = estimates[0]
    comparisons = {}
    for est in estimates[1:]:
        ratio = baseline.estimated_bytes / est.estimated_bytes if est.estimated_bytes > 0 else 0
        savings = baseline.estimated_bytes - est.estimated_bytes
        comparisons[est.dtype.value] = {
            "vs_baseline_ratio": round(ratio, 4),
            "bytes_saved": savings,
            "gb_saved": round(savings / (1024 ** 3), 4),
        }

    return {
        "baseline": baseline.dtype.value,
        "baseline_gb": baseline.estimated_gb,
        "comparisons": comparisons,
    }


def get_storage_summary(estimates: List[StorageEstimate]) -> Dict[str, Any]:
    return {
        "estimates": [e.to_dict() for e in estimates],
        "comparison": compare_storage_estimates(estimates),
    }


def estimate_actual_vs_theoretical(actual_bytes: int, theoretical_bytes: int) -> Dict[str, Any]:
    if theoretical_bytes == 0:
        return {"error": "Theoretical bytes is zero"}

    ratio = actual_bytes / theoretical_bytes
    overhead_bytes = actual_bytes - theoretical_bytes
    overhead_pct = (overhead_bytes / theoretical_bytes) * 100

    return {
        "actual_bytes": actual_bytes,
        "theoretical_bytes": theoretical_bytes,
        "ratio": round(ratio, 4),
        "overhead_bytes": overhead_bytes,
        "overhead_percent": round(overhead_pct, 2),
        "interpretation": (
            "File contains metadata, optimizer states, or multiple copies" if ratio > 1.1
            else "File is compressed or uses efficient serialization" if ratio < 0.9
            else "File size matches theoretical weight storage"
        ),
    }