"""
Mixed-precision search algorithms.

Deterministic strategies for finding feasible configurations under constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from .plan import (
    MixedPrecisionPlan,
    TensorPrecisionAssignment,
    PrecisionType,
    PrecisionCandidate,
    TensorClassification,
)
from .storage import MixedPrecisionStorageModel, StorageAccounting
from .error_analysis import TensorErrorProfile, ErrorMeasurement


@dataclass(frozen=True)
class SearchConstraint:
    """Constraints for mixed-precision search."""
    max_storage_bytes: Optional[int] = None
    max_storage_gb: Optional[float] = None
    max_average_bits_per_weight: Optional[float] = None
    max_mae: Optional[float] = None
    max_relative_error: Optional[float] = None
    min_compression_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_storage_bytes": self.max_storage_bytes,
            "max_storage_gb": self.max_storage_gb,
            "max_average_bits_per_weight": self.max_average_bits_per_weight,
            "max_mae": self.max_mae,
            "max_relative_error": self.max_relative_error,
            "min_compression_ratio": self.min_compression_ratio,
        }


@dataclass(frozen=True)
class Configuration:
    """A mixed-precision configuration with its metrics."""
    plan: MixedPrecisionPlan
    storage: StorageAccounting
    error_metrics: Dict[str, float]  # Aggregated error metrics
    constraint_violations: List[str] = field(default_factory=list)
    is_feasible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "storage": self.storage.to_dict(),
            "error_metrics": self.error_metrics,
            "constraint_violations": self.constraint_violations,
            "is_feasible": self.is_feasible,
        }


@dataclass(frozen=True)
class SearchResult:
    """Result of a mixed-precision search."""
    feasible_configurations: List[Configuration]
    infeasible_configurations: List[Configuration]
    search_metadata: Dict[str, Any]
    
    def get_pareto_frontier(self) -> List[Configuration]:
        """Get Pareto-optimal configurations (storage vs error tradeoff)."""
        feasible = self.feasible_configurations
        if not feasible:
            return []
        
        # Sort by storage (ascending) and MAE (ascending)
        pareto = []
        for config in feasible:
            dominated = False
            storage = config.storage.total_bytes
            mae = config.error_metrics.get("mae", float('inf'))
            
            for other in feasible:
                if other is config:
                    continue
                other_storage = other.storage.total_bytes
                other_mae = other.error_metrics.get("mae", float('inf'))
                
                if other_storage <= storage and other_mae <= mae:
                    if other_storage < storage or other_mae < mae:
                        dominated = True
                        break
            
            if not dominated:
                pareto.append(config)
        
        # Sort by storage
        pareto.sort(key=lambda c: c.storage.total_bytes)
        return pareto
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible_configurations": [c.to_dict() for c in self.feasible_configurations],
            "infeasible_configurations": [c.to_dict() for c in self.infeasible_configurations],
            "search_metadata": self.search_metadata,
            "pareto_frontier": [c.to_dict() for c in self.get_pareto_frontier()],
        }


class MixedPrecisionSearcher:
    """Searches for feasible mixed-precision configurations."""
    
    def __init__(
        self,
        classifications: Dict[str, TensorClassification],
        error_profiles: Dict[str, "TensorErrorProfile"],
        storage_model: MixedPrecisionStorageModel,
        base_plan: Optional[MixedPrecisionPlan] = None,
    ):
        self.classifications = classifications
        self.error_profiles = error_profiles
        self.storage_model = storage_model
        self.base_plan = base_plan
        self.tensor_names = list(classifications.keys())
    
    def evaluate_plan(
        self,
        plan: MixedPrecisionPlan,
        constraints: Optional[SearchConstraint] = None,
    ) -> Configuration:
        """Evaluate a plan against constraints."""
        # Calculate storage
        storage = self.storage_model.calculate_plan_storage(plan, self.classifications)
        
        # Aggregate error metrics
        error_metrics = self._aggregate_errors(plan)
        
        # Check constraints
        violations = []
        is_feasible = True
        
        if constraints:
            if constraints.max_storage_bytes and storage.total_bytes > constraints.max_storage_bytes:
                violations.append(f"Storage {storage.total_bytes} > {constraints.max_storage_bytes}")
                is_feasible = False
            
            if constraints.max_storage_gb:
                max_bytes = int(constraints.max_storage_gb * 1024**3)
                if storage.total_bytes > max_bytes:
                    violations.append(f"Storage {storage.total_bytes/1024**3:.2f}GB > {constraints.max_storage_gb}GB")
                    is_feasible = False
            
            if constraints.max_average_bits_per_weight:
                if storage.average_bits_per_weight > constraints.max_average_bits_per_weight:
                    violations.append(f"Avg BPW {storage.average_bits_per_weight:.3f} > {constraints.max_average_bits_per_weight:.3f}")
                    is_feasible = False
            
            if constraints.max_mae:
                mae = error_metrics.get("mae", float('inf'))
                if mae > constraints.max_mae:
                    violations.append(f"MAE {mae:.6f} > {constraints.max_mae:.6f}")
                    is_feasible = False
            
            if constraints.max_relative_error:
                rel = error_metrics.get("relative_error", float('inf'))
                if rel > constraints.max_relative_error:
                    violations.append(f"Relative error {rel:.6f} > {constraints.max_relative_error:.6f}")
                    is_feasible = False
            
            if constraints.min_compression_ratio:
                if storage.compression_ratio < constraints.min_compression_ratio:
                    violations.append(f"Compression ratio {storage.compression_ratio:.2f} < {constraints.min_compression_ratio:.2f}")
                    is_feasible = False
        
        return Configuration(
            plan=plan,
            storage=storage,
            error_metrics=error_metrics,
            constraint_violations=violations,
            is_feasible=is_feasible,
        )
    
    def _aggregate_errors(self, plan: MixedPrecisionPlan) -> Dict[str, float]:
        """Aggregate error metrics across all tensors in a plan."""
        total_mae = 0.0
        total_mse = 0.0
        total_params = 0
        max_rel_error = 0.0
        
        for tensor_name, assignment in plan.assignments.items():
            profile = self.error_profiles.get(tensor_name)
            if not profile:
                continue
            
            measurement = profile.get_measurement(assignment.precision)
            if not measurement:
                continue
            
            classification = self.classifications.get(tensor_name)
            if not classification:
                continue
            
            weight = classification.num_parameters
            total_params += weight
            total_mae += measurement.mae * weight
            total_mse += measurement.mse * weight
            max_rel_error = max(max_rel_error, measurement.relative_error)
        
        if total_params == 0:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0, "relative_error": 0.0}
        
        return {
            "mae": total_mae / total_params,
            "mse": total_mse / total_params,
            "rmse": np.sqrt(total_mse / total_params),
            "relative_error": max_rel_error,
        }


def greedy_search(
    classifications: Dict[str, TensorClassification],
    error_profiles: Dict[str, "TensorErrorProfile"],
    storage_model: MixedPrecisionStorageModel,
    constraints: SearchConstraint,
    base_plan: Optional[MixedPrecisionPlan] = None,
    precision_order: Optional[List[PrecisionType]] = None,
) -> SearchResult:
    """
    Greedy search for mixed-precision configuration.
    
    Strategy:
    1. Start with base plan (or all FP16)
    2. Iteratively replace tensors with lower precision
    3. Accept substitution if constraints still satisfied
    4. Track all explored configurations
    
    This is a deterministic heuristic, NOT globally optimal.
    """
    if precision_order is None:
        precision_order = [
            PrecisionType.FP16,
            PrecisionType.INT8,
            PrecisionType.INT4,
            PrecisionType.BINARY,
            PrecisionType.TERNARY,
        ]
    
    # Default base plan: all FP16
    if base_plan is None:
        tensor_names = list(classifications.keys())
        base_plan = create_uniform_plan("base", tensor_names, PrecisionType.FP16)
    
    searcher = MixedPrecisionSearcher(
        classifications=classifications,
        error_profiles=error_profiles,
        storage_model=storage_model,
        base_plan=base_plan,
    )
    
    # Evaluate base plan
    feasible = []
    infeasible = []
    
    base_config = searcher.evaluate_plan(base_plan, constraints)
    if base_config.is_feasible:
        feasible.append(base_config)
    else:
        infeasible.append(base_config)
    
    # Try substitutions
    current_plan = base_plan
    visited = {base_plan.experiment_id}
    
    # Sort tensors by potential savings (storage reduction per error increase)
    substitution_candidates = []
    for name, classification in classifications.items():
        current_prec = current_plan.get_precision(name)
        if not current_prec:
            continue
        
        profile = error_profiles.get(name)
        if not profile:
            continue
        
        # Try each lower precision
        current_idx = precision_order.index(current_prec) if current_prec in precision_order else -1
        if current_idx < 0:
            continue
        
        for lower_prec in precision_order[current_idx + 1:]:
            measurement = profile.get_measurement(lower_prec)
            if not measurement:
                continue
            
            # Estimate savings
            storage_model_copy = MixedPrecisionStorageModel()
            current_assignment = current_plan.get_assignment(name)
            if current_assignment:
                current_storage = storage_model_copy.calculate_tensor_storage(current_assignment)
                new_assignment = TensorPrecisionAssignment(
                    tensor_name=name,
                    precision=lower_prec,
                    group_size=measurement.group_size,
                )
                new_storage = storage_model_copy.calculate_tensor_storage(new_assignment, classification)
                
                storage_savings = current_storage.total_bytes - new_storage.total_bytes
                error_increase = measurement.mae - current_assignment.mae if hasattr(current_assignment, 'mae') else measurement.mae
                
                if storage_savings > 0:
                    substitution_candidates.append({
                        "tensor_name": name,
                        "from_precision": current_prec,
                        "to_precision": lower_prec,
                        "storage_savings": storage_savings,
                        "error_increase": error_increase,
                        "efficiency": storage_savings / max(error_increase, 1e-9),
                    })
    
    # Sort by efficiency (savings per error increase)
    substitution_candidates.sort(key=lambda x: x["efficiency"], reverse=True)
    
    # Apply substitutions greedily
    for candidate in substitution_candidates:
        name = candidate["tensor_name"]
        new_prec = candidate["to_precision"]
        
        # Create new plan with substitution
        new_assignments = dict(current_plan.assignments)
        new_assignments[name] = TensorPrecisionAssignment(
            tensor_name=name,
            precision=new_prec,
            group_size=candidate["group_size"] if "group_size" in candidate else 128,
        )
        
        new_plan = MixedPrecisionPlan(
            model_identifier=current_plan.model_identifier,
            assignments=new_assignments,
            global_constraints=current_plan.global_constraints,
            experiment_id=str(uuid.uuid4())[:8],
        )
        
        # Evaluate
        config = searcher.evaluate_plan(new_plan, constraints)
        
        if config.is_feasible:
            feasible.append(config)
            current_plan = new_plan
        else:
            infeasible.append(config)
    
    search_metadata = {
        "algorithm": "greedy",
        "precision_order": [p.value for p in precision_order],
        "total_explored": len(feasible) + len(infeasible),
        "feasible_count": len(feasible),
        "infeasible_count": len(infeasible),
        "constraints": constraints.to_dict(),
    }
    
    return SearchResult(
        feasible_configurations=feasible,
        infeasible_configurations=infeasible,
        search_metadata=search_metadata,
    )


def exhaustive_search(
    classifications: Dict[str, TensorClassification],
    error_profiles: Dict[str, "TensorErrorProfile"],
    storage_model: MixedPrecisionStorageModel,
    constraints: SearchConstraint,
    tensor_precision_limits: Optional[Dict[str, List[PrecisionType]]] = None,
) -> SearchResult:
    """
    Exhaustive search (for small models only).
    
    Tries all combinations of allowed precisions for each tensor.
    Only practical for models with very few tensors.
    """
    searcher = MixedPrecisionSearcher(
        classifications=classifications,
        error_profiles=error_profiles,
        storage_model=storage_model,
    )
    
    # Default: allow all quantized precisions
    if tensor_precision_limits is None:
        tensor_precision_limits = {}
        for name in classifications:
            tensor_precision_limits[name] = [
                PrecisionType.FP16,
                PrecisionType.INT8,
                PrecisionType.INT4,
                PrecisionType.BINARY,
                PrecisionType.TERNARY,
            ]
    
    tensor_names = list(classifications.keys())
    
    feasible = []
    infeasible = []
    
    def explore(idx: int, current_assignments: Dict[str, TensorPrecisionAssignment]):
        if idx >= len(tensor_names):
            # Evaluate complete plan
            plan = MixedPrecisionPlan(
                model_identifier="exhaustive",
                assignments=current_assignments,
            )
            config = searcher.evaluate_plan(plan, constraints)
            if config.is_feasible:
                feasible.append(config)
            else:
                infeasible.append(config)
            return
        
        name = tensor_names[idx]
        allowed = tensor_precision_limits.get(name, [PrecisionType.FP16])
        
        for prec in allowed:
            profile = error_profiles.get(name)
            if not profile or prec not in profile.measurements:
                # Skip if no measurement available
                if prec == PrecisionType.FP16:
                    assignment = TensorPrecisionAssignment(
                        tensor_name=name,
                        precision=prec,
                    )
                    new_assignments = {**current_assignments, name: assignment}
                    explore(idx + 1, new_assignments)
                continue
            
            measurement = profile.get_measurement(prec)
            if not measurement:
                continue
            
            assignment = TensorPrecisionAssignment(
                tensor_name=name,
                precision=prec,
                group_size=measurement.group_size,
            )
            new_assignments = {**current_assignments, name: assignment}
            explore(idx + 1, new_assignments)
    
    explore(0, {})
    
    search_metadata = {
        "algorithm": "exhaustive",
        "total_explored": len(feasible) + len(infeasible),
        "feasible_count": len(feasible),
        "infeasible_count": len(infeasible),
        "constraints": constraints.to_dict(),
    }
    
    return SearchResult(
        feasible_configurations=feasible,
        infeasible_configurations=infeasible,
        search_metadata=search_metadata,
    )


# Need to import uuid
import uuid