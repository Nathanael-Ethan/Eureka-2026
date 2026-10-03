"""
Export functionality for mixed-precision experiments.
"""

import csv
import json
import os
from typing import Dict, List, Any
from .plan import MixedPrecisionPlan, TensorPrecisionAssignment, PrecisionType
from .storage import StorageAccounting, StorageComponent
from .search import SearchResult, Configuration
from .scenarios import ScenarioResult
from .error_analysis import TensorErrorProfile


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dictionary for CSV export."""
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, f"{new_key}_"))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items


def export_plan_csv(plan: MixedPrecisionPlan, filepath: str) -> None:
    """Export a mixed-precision plan to CSV."""
    rows = []
    for name, assignment in plan.assignments.items():
        row = {
            "tensor_name": name,
            "precision": assignment.precision.value,
            "group_size": assignment.group_size,
            "codebook_size": assignment.codebook_size,
        }
        if assignment.candidate:
            row.update(_flatten_dict(assignment.candidate.to_dict(), "candidate_"))
        rows.append(row)
    
    if not rows:
        return
    
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_plan_json(plan: MixedPrecisionPlan, filepath: str, indent: int = 2) -> None:
    """Export a mixed-precision plan to JSON."""
    data = plan.to_dict()
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def export_storage_accounting_csv(storage: StorageAccounting, filepath: str) -> None:
    """Export storage accounting to CSV."""
    rows = []
    for comp in storage.components:
        row = comp.to_dict()
        rows.append(row)
    
    if not rows:
        return
    
    # Add summary row
    summary = {
        "tensor_name": "TOTAL",
        "precision": "",
        "weight_bytes": sum(c.weight_bytes for c in storage.components),
        "scale_bytes": sum(c.scale_bytes for c in storage.components),
        "metadata_bytes": sum(c.metadata_bytes for c in storage.components),
        "total_bytes": storage.total_bytes,
        "bits_per_weight": storage.average_bits_per_weight,
        "group_size": "",
        "codebook_size": "",
    }
    rows.append(summary)
    
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_search_results_csv(result: SearchResult, filepath: str) -> None:
    """Export search results to CSV."""
    rows = []
    
    for config in result.feasible_configurations + result.infeasible_configurations:
        row = {
            "feasible": config.is_feasible,
            "total_bytes": config.storage.total_bytes,
            "avg_bpw": config.storage.average_bits_per_weight,
            "compression_ratio": config.storage.compression_ratio,
            "mae": config.error_metrics.get("mae", 0),
            "mse": config.error_metrics.get("mse", 0),
            "relative_error": config.error_metrics.get("relative_error", 0),
            "constraint_violations": "; ".join(config.constraint_violations),
            "precision_distribution": str(config.plan.to_dict().get("precision_distribution", {})),
        }
        rows.append(row)
    
    if not rows:
        return
    
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_search_results_json(result: SearchResult, filepath: str, indent: int = 2) -> None:
    """Export search results to JSON."""
    data = result.to_dict()
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def export_scenario_results_csv(result: ScenarioResult, filepath: str) -> None:
    """Export scenario results to CSV."""
    rows = []
    
    # Uniform results
    for prec, storage in result.uniform_storage.items():
        row = {
            "scenario": result.scenario.name,
            "strategy": f"uniform_{prec.value}",
            "total_bytes": storage.total_bytes,
            "avg_bpw": storage.average_bits_per_weight,
            "compression_ratio": storage.compression_ratio,
            "savings_vs_fp16_pct": 0.0,
        }
        rows.append(row)
    
    # Mixed results
    fp16_bytes = result.uniform_storage.get(PrecisionType.FP16)
    fp16_bytes = fp16_bytes.total_bytes if fp16_bytes else 1
    
    for strat_name, storage in result.mixed_storage.items():
        row = {
            "scenario": result.scenario.name,
            "strategy": strat_name,
            "total_bytes": storage.total_bytes,
            "avg_bpw": storage.average_bits_per_weight,
            "compression_ratio": storage.compression_ratio,
            "savings_vs_fp16_pct": result.storage_savings.get(strat_name, 0),
        }
        rows.append(row)
    
    if not rows:
        return
    
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_scenario_results_json(result: ScenarioResult, filepath: str, indent: int = 2) -> None:
    """Export scenario results to JSON."""
    data = result.to_dict()
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def export_error_profiles_csv(
    profiles: Dict[str, TensorErrorProfile],
    filepath: str,
) -> None:
    """Export error profiles to CSV."""
    rows = []
    for name, profile in profiles.items():
        for prec, meas in profile.measurements.items():
            row = {
                "tensor_name": name,
                "role": profile.tensor_classification.role.value,
                "precision": prec.value,
                **meas.to_dict(),
            }
            rows.append(row)
    
    if not rows:
        return
    
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_plan_json(filepath: str) -> MixedPrecisionPlan:
    """Load a mixed-precision plan from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assignments = {}
    for name, a in data.get("assignments", {}).items():
        candidate_data = a.get("candidate")
        candidate = None
        if candidate_data:
            candidate = PrecisionCandidate(
                precision=PrecisionType(candidate_data["precision"]),
                group_size=candidate_data.get("group_size"),
                codebook_size=candidate_data.get("codebook_size"),
                estimated_bits_per_weight=candidate_data.get("estimated_bits_per_weight", 0),
                estimated_storage_bytes=candidate_data.get("estimated_storage_bytes", 0),
            )
        assignments[name] = TensorPrecisionAssignment(
            tensor_name=name,
            precision=PrecisionType(a["precision"]),
            group_size=a.get("group_size"),
            codebook_size=a.get("codebook_size"),
            candidate=candidate,
        )
    
    return MixedPrecisionPlan(
        model_identifier=data.get("model_identifier", "loaded"),
        assignments=assignments,
        global_constraints=data.get("global_constraints", {}),
        experiment_id=data.get("experiment_id", ""),
        timestamp=data.get("timestamp", ""),
        notes=data.get("notes", ""),
    )