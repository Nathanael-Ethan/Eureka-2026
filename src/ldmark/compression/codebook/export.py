"""
Export functionality for codebook quantization experiments.
"""

import csv
import json
from typing import List, Dict, Any
import os
import numpy as np

from .experiment import CodebookExperimentResult
from .analysis import WeightDistributionStats


def _convert_to_native(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_native(v) for v in obj)
    else:
        return obj


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionary for CSV export."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, f"{new_key}_"))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            # Don't flatten lists of dicts, just serialize
            items[new_key] = json.dumps(_convert_to_native(v))
        else:
            items[new_key] = _convert_to_native(v)
    return items


def export_experiments_csv(
    results: List[CodebookExperimentResult],
    filepath: str,
) -> None:
    """
    Export codebook experiment results to CSV.
    
    Args:
        results: List of CodebookExperimentResult
        filepath: Output CSV file path
    """
    if not results:
        return
    
    rows = []
    for r in results:
        row = _flatten_dict(r.to_dict())
        rows.append(row)
    
    # Collect all fieldnames
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_experiments_json(
    results: List[CodebookExperimentResult],
    filepath: str,
    indent: int = 2,
) -> None:
    """
    Export codebook experiment results to JSON.
    
    Args:
        results: List of CodebookExperimentResult
        filepath: Output JSON file path
        indent: JSON indentation
    """
    data = [_convert_to_native(r.to_dict()) for r in results]
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def export_comparison_csv(
    results: List[CodebookExperimentResult],
    filepath: str,
) -> None:
    """
    Export comparison table (codebook vs uniform) to CSV.
    
    Creates a table with one row per experiment and columns for
    both codebook and uniform metrics.
    """
    if not results:
        return
    
    rows = []
    for r in results:
        if r.codebook_error is None or r.uniform_error is None:
            continue
            
        row = {
            "experiment_id": r.experiment_id,
            "experiment_name": r.config.experiment_name,
            "tensor_shape": str(r.config.tensor_shape),
            "source_dtype": str(r.config.source_dtype),
            "codebook_size": r.config.codebook_size,
            "group_size": r.config.group_size,
            "initialization": r.config.initialization,
            "tensor_generator": r.config.tensor_generator,
            "seed": r.config.seed,
            
            # Codebook metrics
            "cb_mae": r.codebook_error["mae"],
            "cb_mse": r.codebook_error["mse"],
            "cb_rmse": r.codebook_error["rmse"],
            "cb_max_error": r.codebook_error["max_absolute_error"],
            "cb_relative_error": r.codebook_error["relative_error"],
            "cb_storage_bytes": r.codebook_storage.total_bytes,
            "cb_effective_bpw": r.codebook_storage.effective_bits_per_weight,
            "cb_compression_ratio": r.codebook_storage.compression_ratio,
            "cb_index_bytes": r.codebook_storage.index_bytes,
            "cb_codebook_bytes": r.codebook_storage.codebook_bytes,
            
            # Uniform INT4 metrics
            "u_mae": r.uniform_error["mae"],
            "u_mse": r.uniform_error["mse"],
            "u_rmse": r.uniform_error["rmse"],
            "u_max_error": r.uniform_error["max_absolute_error"],
            "u_relative_error": r.uniform_error["relative_error"],
            "u_storage_bytes": r.uniform_storage["total_bytes"],
            "u_bits_per_weight": r.uniform_storage["bits_per_weight"],
            "u_compression_ratio": r.uniform_storage["compression_ratio"],
            
            # Differences
            "mae_diff": r.codebook_error["mae"] - r.uniform_error["mae"],
            "mse_diff": r.codebook_error["mse"] - r.uniform_error["mse"],
            "max_error_diff": r.codebook_error["max_absolute_error"] - r.uniform_error["max_absolute_error"],
            "rel_error_diff": r.codebook_error["relative_error"] - r.uniform_error["relative_error"],
            "bpw_diff": r.codebook_storage.effective_bits_per_weight - r.uniform_storage["bits_per_weight"],
            "storage_diff_bytes": r.codebook_storage.total_bytes - r.uniform_storage["total_bytes"],
            "mae_ratio": r.codebook_error["mae"] / r.uniform_error["mae"] if r.uniform_error["mae"] > 0 else 0,
            
            # Weight distribution
            "weight_mean": r.weight_distribution.mean,
            "weight_std": r.weight_distribution.std,
            "weight_skew": r.weight_distribution.skew,
            "weight_kurtosis": r.weight_distribution.kurtosis,
            
            # Timing
            "generation_time_ms": r.generation_time_ms,
            "total_time_ms": r.total_time_ms,
        }
        rows.append(row)
    
    if not rows:
        return
    
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_binary_comparison_csv(
    binary_results: List[Dict[str, Any]],
    filepath: str,
) -> None:
    """
    Export binary vs learned 2-value codebook comparison to CSV.
    """
    if not binary_results:
        return
    
    rows = []
    for r in binary_results:
        fixed = r["fixed_binary"]
        learned = r["learned_binary"]
        imp = r["improvement"]
        
        row = {
            "shape": str(r["shape"]),
            "generator": r["generator"],
            
            "fixed_mae": fixed["mae"],
            "fixed_mse": fixed["mse"],
            "fixed_max_error": fixed["max_abs_error"],
            "fixed_rel_error": fixed["relative_error"],
            "fixed_storage_bytes": fixed["storage_bytes"],
            "fixed_bpw": fixed["bpw"],
            
            "learned_mae": learned["mae"],
            "learned_mse": learned["mse"],
            "learned_max_error": learned["max_abs_error"],
            "learned_rel_error": learned["relative_error"],
            "learned_storage_bytes": learned["storage_bytes"],
            "learned_bpw": learned["bpw"],
            
            "mae_reduction_pct": imp["mae_reduction"] * 100,
            "mse_reduction_pct": imp["mse_reduction"] * 100,
            "max_error_reduction_pct": imp["max_error_reduction"] * 100,
            
            "learned_codebooks": str(learned["codebooks"]),
        }
        rows.append(row)
    
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_experiments_json(filepath: str) -> List[CodebookExperimentResult]:
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [CodebookExperimentResult(**d) for d in data]