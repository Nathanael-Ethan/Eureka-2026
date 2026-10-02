"""
Export functionality for LDMARK benchmark results.

Provides CSV and JSON export suitable for later visualization and analysis.
"""

import csv
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from .model import (
    BenchmarkRecord,
    TensorBenchmarkResult,
    ModelBenchmarkResult,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
)


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionary for CSV export."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, f"{new_key}_"))
        else:
            items[new_key] = v
    return items


def export_benchmarks_csv(
    records: List[BenchmarkRecord],
    filepath: str,
    include_error_distribution: bool = True,
    include_timing: bool = True,
    include_storage: bool = True,
    include_hardware: bool = False,
) -> None:
    """
    Export benchmark records to CSV file.
    
    Creates a flat table with one row per benchmark record.
    Suitable for loading into pandas, spreadsheets, or plotting tools.
    
    Args:
        records: List of BenchmarkRecord objects
        filepath: Output CSV file path
        include_error_distribution: Include min/median/mean/max error stats
        include_timing: Include transformation/dequantization/validation times
        include_storage: Include theoretical/actual storage breakdown
        include_hardware: Include hardware metadata
    """
    if not records:
        return
    
    # Prepare rows
    rows = []
    for record in records:
        row = {
            "experiment_id": record.experiment_id,
            "timestamp": record.timestamp,
            "model_identifier": record.model_identifier,
            "parameter_count": record.parameter_count,
            "tensor_count": record.tensor_count,
            "source_dtype": record.source_dtype,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "scale_dtype": record.scale_dtype,
            "original_bytes": record.original_bytes,
            "compressed_bytes": record.compressed_bytes,
            "actual_bits_per_weight": record.actual_bits_per_weight,
            "compression_ratio": record.compression_ratio,
            "mae": record.mae,
            "mse": record.mse,
            "rmse": record.rmse,
            "max_absolute_error": record.max_absolute_error,
            "relative_error": record.relative_error,
            "transformation_time_ms": record.transformation_time_ms,
            "dequantization_time_ms": record.dequantization_time_ms,
            "validation_time_ms": record.validation_time_ms,
            "total_time_ms": record.total_time_ms,
            "theoretical_storage_bytes": record.theoretical_storage_bytes,
            "actual_storage_bytes": record.actual_storage_bytes,
            "metadata_bytes": record.metadata_bytes,
            "scales_bytes": record.scales_bytes,
            "padding_bytes": record.padding_bytes,
            "tensor_headers_bytes": record.tensor_headers_bytes,
            "serialization_overhead_bytes": record.serialization_overhead_bytes,
            "random_seed": record.random_seed,
            "notes": record.notes,
        }
        
        if include_error_distribution:
            row.update({
                "error_min": record.error_distribution.minimum,
                "error_median": record.error_distribution.median,
                "error_mean": record.error_distribution.mean,
                "error_max": record.error_distribution.maximum,
                "error_std": record.error_distribution.std,
                "error_p25": record.error_distribution.p25,
                "error_p75": record.error_distribution.p75,
                "error_p95": record.error_distribution.p95,
                "error_p99": record.error_distribution.p99,
                "error_count": record.error_distribution.count,
            })
        
        if include_hardware:
            hw = record.hardware_metadata
            row.update({
                "hw_platform": hw.platform,
                "hw_processor": hw.processor,
                "hw_python_version": hw.python_version,
                "hw_numpy_version": hw.numpy_version,
                "hw_cpu_count": hw.cpu_count,
                "hw_total_memory_gb": hw.total_memory_gb,
            })
        
        rows.append(row)
    
    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_benchmarks_json(
    records: List[BenchmarkRecord],
    filepath: str,
    indent: int = 2,
) -> None:
    """
    Export benchmark records to JSON file.
    
    Preserves full nested structure including error distributions,
    timing metrics, and storage metrics.
    
    Args:
        records: List of BenchmarkRecord objects
        filepath: Output JSON file path
        indent: JSON indentation level
    """
    data = [record.to_dict() for record in records]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def export_tensor_results_csv(
    results: List[TensorBenchmarkResult],
    filepath: str,
) -> None:
    """
    Export tensor-level benchmark results to CSV.
    
    One row per tensor per experiment.
    """
    if not results:
        return
    
    rows = []
    for r in results:
        row = {
            "tensor_name": r.tensor_name,
            "original_shape": str(r.original_shape),
            "original_dtype": r.original_dtype,
            "num_elements": r.num_elements,
            "compression_method": r.compression_method,
            "target_bits": r.target_bits,
            "group_size": r.group_size,
            "scale_dtype": r.scale_dtype,
            "original_bytes": r.original_bytes,
            "compressed_bytes": r.compressed_bytes,
            "actual_bits_per_weight": r.actual_bits_per_weight,
            "compression_ratio": r.compression_ratio,
            "relative_error": r.relative_error,
            "error_min": r.error_distribution.minimum,
            "error_median": r.error_distribution.median,
            "error_mean": r.error_distribution.mean,
            "error_max": r.error_distribution.maximum,
            "error_std": r.error_distribution.std,
            "error_p25": r.error_distribution.p25,
            "error_p75": r.error_distribution.p75,
            "error_p95": r.error_distribution.p95,
            "error_p99": r.error_distribution.p99,
            "transform_time_ms": r.timing.transformation_time_ms,
            "dequant_time_ms": r.timing.dequantization_time_ms,
            "validation_time_ms": r.timing.validation_time_ms,
            "total_time_ms": r.timing.total_time_ms,
            "theoretical_storage_bytes": r.storage.theoretical_bytes,
            "actual_storage_bytes": r.storage.actual_bytes,
            "scales_bytes": r.storage.scales_bytes,
            "overhead_ratio": r.storage.overhead_ratio,
            "random_seed": r.random_seed,
        }
        rows.append(row)
    
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_model_results_csv(
    results: List[ModelBenchmarkResult],
    filepath: str,
) -> None:
    """
    Export model-level benchmark results to CSV.
    
    One row per model per experiment.
    """
    if not results:
        return
    
    rows = []
    for r in results:
        row = {
            "model_identifier": r.model_identifier,
            "parameter_count": r.parameter_count,
            "tensor_count": r.tensor_count,
            "aggregate_compression_ratio": r.aggregate_compression_ratio,
            "aggregate_bits_per_weight": r.aggregate_bits_per_weight,
            "aggregate_original_bytes": r.aggregate_original_bytes,
            "aggregate_compressed_bytes": r.aggregate_compressed_bytes,
            "total_transform_time_ms": r.total_timing.transformation_time_ms,
            "total_dequant_time_ms": r.total_timing.dequantization_time_ms,
            "total_validation_time_ms": r.total_timing.validation_time_ms,
            "total_time_ms": r.total_timing.total_time_ms,
            "total_theoretical_storage_bytes": r.aggregate_storage.theoretical_bytes,
            "total_actual_storage_bytes": r.aggregate_storage.actual_bytes,
            "total_scales_bytes": r.aggregate_storage.scales_bytes,
            "total_metadata_bytes": r.aggregate_storage.metadata_bytes,
            "total_padding_bytes": r.aggregate_storage.padding_bytes,
            "total_tensor_headers_bytes": r.aggregate_storage.tensor_headers_bytes,
            "total_overhead_ratio": r.aggregate_storage.overhead_ratio,
            "random_seed": r.random_seed,
        }
        rows.append(row)
    
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_error_distributions_csv(
    records: List[BenchmarkRecord],
    filepath: str,
) -> None:
    """
    Export error distributions to CSV for visualization.
    
    Creates long-format data suitable for violin plots, box plots, histograms.
    Each row represents one error statistic for one experiment.
    """
    rows = []
    for record in records:
        dist = record.error_distribution
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "min",
            "value": dist.minimum,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "median",
            "value": dist.median,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "mean",
            "value": dist.mean,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "max",
            "value": dist.maximum,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "std",
            "value": dist.std,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "p25",
            "value": dist.p25,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "p75",
            "value": dist.p75,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "p95",
            "value": dist.p95,
        })
        rows.append({
            "experiment_id": record.experiment_id,
            "model_identifier": record.model_identifier,
            "compression_method": record.compression_method,
            "target_bits": record.target_bits,
            "group_size": record.group_size,
            "statistic": "p99",
            "value": dist.p99,
        })
    
    fieldnames = ["experiment_id", "model_identifier", "compression_method", "target_bits", "group_size", "statistic", "value"]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_comparison_table_csv(
    records: List[BenchmarkRecord],
    filepath: str,
    group_by: List[str] = None,
) -> None:
    """
    Export method comparison table to CSV.
    
    Creates a pivot table with methods as columns and metrics as rows.
    """
    if group_by is None:
        group_by = ["model_identifier", "target_bits", "group_size"]
    
    # Group records
    groups: Dict[tuple, List[BenchmarkRecord]] = {}
    for record in records:
        key = tuple(getattr(record, k) for k in group_by)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    
    # Build comparison rows
    rows = []
    for key, group_records in groups.items():
        group_dict = dict(zip(group_by, key))
        
        # Find each method's metrics
        methods = {}
        for r in group_records:
            methods[r.compression_method] = r
        
        row = group_dict.copy()
        for method_name, method_record in methods.items():
            row[f"{method_name}_bits_per_weight"] = method_record.actual_bits_per_weight
            row[f"{method_name}_compression_ratio"] = method_record.compression_ratio
            row[f"{method_name}_mae"] = method_record.mae
            row[f"{method_name}_mse"] = method_record.mse
            row[f"{method_name}_rmse"] = method_record.rmse
            row[f"{method_name}_max_error"] = method_record.max_absolute_error
            row[f"{method_name}_relative_error"] = method_record.relative_error
            row[f"{method_name}_original_bytes"] = method_record.original_bytes
            row[f"{method_name}_compressed_bytes"] = method_record.compressed_bytes
            row[f"{method_name}_theoretical_storage"] = method_record.theoretical_storage_bytes
            row[f"{method_name}_actual_storage"] = method_record.actual_storage_bytes
            row[f"{method_name}_overhead_ratio"] = method_record.actual_storage_bytes / method_record.theoretical_storage_bytes if method_record.theoretical_storage_bytes > 0 else 0
            row[f"{method_name}_transform_time_ms"] = method_record.transformation_time_ms
            row[f"{method_name}_dequant_time_ms"] = method_record.dequantization_time_ms
        
        rows.append(row)
    
    if rows:
        # Collect all possible fieldnames from all rows
        fieldnames = set()
        for row in rows:
            fieldnames.update(row.keys())
        fieldnames = list(fieldnames)
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_benchmarks_json(filepath: str) -> List[BenchmarkRecord]:
    """Load benchmark records from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [BenchmarkRecord.from_dict(d) for d in data]


def load_benchmarks_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark records from CSV file as list of dicts."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)