"""
Model-scale scenarios for mixed-precision analysis.

Creates analytical scenarios for 7B, 13B, 27B parameter models
using parameter distributions from known architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from .plan import (
    MixedPrecisionPlan,
    TensorPrecisionAssignment,
    PrecisionType,
    PrecisionCandidate,
    TensorClassification,
    TensorRole,
)
from .storage import MixedPrecisionStorageModel, StorageAccounting
from .error_analysis import TensorErrorProfile, ErrorMeasurement


@dataclass(frozen=True)
class ModelScaleScenario:
    """Scenario definition for a model scale."""
    name: str
    total_parameters: int
    architecture: str
    layer_count: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    tensor_distribution: Dict[TensorRole, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_parameters": self.total_parameters,
            "architecture": self.architecture,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "vocab_size": self.vocab_size,
            "tensor_distribution": {
                k.value: v for k, v in self.tensor_distribution.items()
            },
        }


@dataclass(frozen=True)
class ScenarioResult:
    """Result of a model-scale scenario analysis."""
    scenario: ModelScaleScenario
    uniform_storage: Dict[PrecisionType, StorageAccounting]
    mixed_storage: Dict[str, StorageAccounting]  # strategy name -> storage
    mixed_plans: Dict[str, MixedPrecisionPlan]  # strategy name -> plan
    storage_savings: Dict[str, float]  # strategy -> savings vs uniform FP16
    parameter_distribution: Dict[TensorRole, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "uniform_storage": {
                p.value: s.to_dict() for p, s in self.uniform_storage.items()
            },
            "mixed_storage": {
                k: s.to_dict() for k, s in self.mixed_storage.items()
            },
            "mixed_plans": {
                k: p.to_dict() for k, p in self.mixed_plans.items()
            },
            "storage_savings": self.storage_savings,
            "parameter_distribution": {
                k.value: v for k, v in self.parameter_distribution.items()
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


def get_default_scenarios() -> List[ModelScaleScenario]:
    """Get default model scale scenarios (7B, 13B, 27B)."""
    return [
        # 7B model (LLaMA-7B like)
        ModelScaleScenario(
            name="7B",
            total_parameters=6_700_000_000,
            architecture="llama",
            layer_count=32,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            vocab_size=32000,
            tensor_distribution={
                TensorRole.EMBEDDING: {"count": 2, "params_per": 32000 * 4096},
                TensorRole.ATTENTION_Q: {"count": 32, "params_per": 4096 * 4096},
                TensorRole.ATTENTION_K: {"count": 32, "params_per": 4096 * 4096},
                TensorRole.ATTENTION_V: {"count": 32, "params_per": 4096 * 4096},
                TensorRole.ATTENTION_O: {"count": 32, "params_per": 4096 * 4096},
                TensorRole.MLP_UP: {"count": 32, "params_per": 4096 * 11008},
                TensorRole.MLP_DOWN: {"count": 32, "params_per": 11008 * 4096},
                TensorRole.NORM: {"count": 32 * 2 + 2, "params_per": 4096},
                TensorRole.OUTPUT_HEAD: {"count": 1, "params_per": 4096 * 32000},
            },
        ),
        # 13B model (LLaMA-13B like)
        ModelScaleScenario(
            name="13B",
            total_parameters=13_000_000_000,
            architecture="llama",
            layer_count=40,
            hidden_size=5120,
            intermediate_size=13824,
            num_attention_heads=40,
            vocab_size=32000,
            tensor_distribution={
                TensorRole.EMBEDDING: {"count": 2, "params_per": 32000 * 5120},
                TensorRole.ATTENTION_Q: {"count": 40, "params_per": 5120 * 5120},
                TensorRole.ATTENTION_K: {"count": 40, "params_per": 5120 * 5120},
                TensorRole.ATTENTION_V: {"count": 40, "params_per": 5120 * 5120},
                TensorRole.ATTENTION_O: {"count": 40, "params_per": 5120 * 5120},
                TensorRole.MLP_UP: {"count": 40, "params_per": 5120 * 13824},
                TensorRole.MLP_DOWN: {"count": 40, "params_per": 13824 * 5120},
                TensorRole.NORM: {"count": 40 * 2 + 2, "params_per": 5120},
                TensorRole.OUTPUT_HEAD: {"count": 1, "params_per": 5120 * 32000},
            },
        ),
        # 27B model (PrismML scale)
        ModelScaleScenario(
            name="27B",
            total_parameters=27_000_000_000,
            architecture="llama",
            layer_count=48,
            hidden_size=6144,
            intermediate_size=16640,
            num_attention_heads=48,
            vocab_size=32000,
            tensor_distribution={
                TensorRole.EMBEDDING: {"count": 2, "params_per": 32000 * 6144},
                TensorRole.ATTENTION_Q: {"count": 48, "params_per": 6144 * 6144},
                TensorRole.ATTENTION_K: {"count": 48, "params_per": 6144 * 6144},
                TensorRole.ATTENTION_V: {"count": 48, "params_per": 6144 * 6144},
                TensorRole.ATTENTION_O: {"count": 48, "params_per": 6144 * 6144},
                TensorRole.MLP_UP: {"count": 48, "params_per": 6144 * 16640},
                TensorRole.MLP_DOWN: {"count": 48, "params_per": 16640 * 6144},
                TensorRole.NORM: {"count": 48 * 2 + 2, "params_per": 6144},
                TensorRole.OUTPUT_HEAD: {"count": 1, "params_per": 6144 * 32000},
            },
        ),
        # 70B model
        ModelScaleScenario(
            name="70B",
            total_parameters=70_000_000_000,
            architecture="llama",
            layer_count=80,
            hidden_size=8192,
            intermediate_size=22016,
            num_attention_heads=64,
            vocab_size=32000,
            tensor_distribution={
                TensorRole.EMBEDDING: {"count": 2, "params_per": 32000 * 8192},
                TensorRole.ATTENTION_Q: {"count": 80, "params_per": 8192 * 8192},
                TensorRole.ATTENTION_K: {"count": 80, "params_per": 8192 * 8192},
                TensorRole.ATTENTION_V: {"count": 80, "params_per": 8192 * 8192},
                TensorRole.ATTENTION_O: {"count": 80, "params_per": 8192 * 8192},
                TensorRole.MLP_UP: {"count": 80, "params_per": 8192 * 22016},
                TensorRole.MLP_DOWN: {"count": 80, "params_per": 22016 * 8192},
                TensorRole.NORM: {"count": 80 * 2 + 2, "params_per": 8192},
                TensorRole.OUTPUT_HEAD: {"count": 1, "params_per": 8192 * 32000},
            },
        ),
    ]


def estimate_parameter_distribution(scenario: ModelScaleScenario) -> Dict[TensorRole, int]:
    """Estimate parameter counts per role from scenario."""
    distribution = {}
    for role, info in scenario.tensor_distribution.items():
        count = info.get("count", 0)
        params_per = info.get("params_per", 0)
        distribution[role] = count * params_per
    return distribution


def create_classifications_from_scenario(
    scenario: ModelScaleScenario,
) -> Dict[str, TensorClassification]:
    """Create synthetic tensor classifications from scenario."""
    classifications = {}
    param_dist = estimate_parameter_distribution(scenario)
    
    for role, total_params in param_dist.items():
        count = scenario.tensor_distribution[role].get("count", 1)
        params_per = scenario.tensor_distribution[role].get("params_per", total_params)
        
        # Estimate shape
        if role == TensorRole.EMBEDDING:
            shape = (scenario.vocab_size, scenario.hidden_size)
        elif role == TensorRole.OUTPUT_HEAD:
            shape = (scenario.hidden_size, scenario.vocab_size)
        elif role == TensorRole.NORM:
            shape = (scenario.hidden_size,)
        elif role in (TensorRole.ATTENTION_Q, TensorRole.ATTENTION_K, 
                      TensorRole.ATTENTION_V, TensorRole.ATTENTION_O):
            shape = (scenario.hidden_size, scenario.hidden_size)
        elif role in (TensorRole.MLP_UP, TensorRole.MLP_DOWN):
            shape = (scenario.hidden_size, scenario.intermediate_size) if role == TensorRole.MLP_UP \
                    else (scenario.intermediate_size, scenario.hidden_size)
        else:
            shape = (scenario.hidden_size,)
        
        # Distribute params across instances
        for i in range(count):
            name = f"{role.value}.{i}"
            if role == TensorRole.EMBEDDING:
                name = f"embed_tokens.{i}" if i > 0 else "embed_tokens"
            elif role == TensorRole.NORM:
                name = f"layers.{i}.norm"
            elif role in (TensorRole.ATTENTION_Q, TensorRole.ATTENTION_K,
                         TensorRole.ATTENTION_V, TensorRole.ATTENTION_O):
                layer_idx = i // 4
                attn_type = ["q", "k", "v", "o"][i % 4]
                name = f"layers.{layer_idx}.attn.{attn_type}_proj"
            elif role in (TensorRole.MLP_UP, TensorRole.MLP_DOWN):
                layer_idx = i // 2
                mlp_type = ["up", "down"][i % 2]
                name = f"layers.{layer_idx}.mlp.{mlp_type}_proj"
            
            classifications[name] = TensorClassification(
                tensor_name=name,
                role=role,
                shape=shape,
                num_parameters=params_per,
                dtype="float32",
                layer_index=layer_idx if role != TensorRole.EMBEDDING and role != TensorRole.OUTPUT_HEAD else None,
                architecture=scenario.architecture,
                is_weight=True,
            )
    
    return classifications


def create_uniform_fp16_plan(classifications: Dict[str, TensorClassification]) -> MixedPrecisionPlan:
    """Create a uniform FP16 plan as baseline."""
    from .plan import create_uniform_plan
    return create_uniform_plan("uniform_fp16", list(classifications.keys()), PrecisionType.FP16)


def create_mixed_precision_strategies(
    classifications: Dict[str, TensorClassification],
) -> Dict[str, MixedPrecisionPlan]:
    """Create predefined mixed-precision strategies."""
    strategies = {}
    
    # Strategy 1: Conservative - only MLP down-projection to INT4
    assignments = {}
    for name, cls in classifications.items():
        if cls.role == TensorRole.MLP_DOWN:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT4,
                group_size=128,
            )
        elif cls.role == TensorRole.NORM:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
        elif cls.role in (TensorRole.ATTENTION_Q, TensorRole.ATTENTION_K,
                         TensorRole.ATTENTION_V, TensorRole.ATTENTION_O):
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        elif cls.role == TensorRole.EMBEDDING:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        elif cls.role == TensorRole.OUTPUT_HEAD:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        else:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
    strategies["conservative"] = MixedPrecisionPlan(
        model_identifier="mixed_conservative",
        assignments=assignments,
    )
    
    # Strategy 2: Aggressive - all attention + MLP to INT4
    assignments = {}
    for name, cls in classifications.items():
        if cls.role in (TensorRole.ATTENTION_Q, TensorRole.ATTENTION_K,
                       TensorRole.ATTENTION_V, TensorRole.ATTENTION_O,
                       TensorRole.MLP_UP, TensorRole.MLP_DOWN,
                       TensorRole.MLP_GATE):
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT4,
                group_size=128,
            )
        elif cls.role == TensorRole.NORM:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
        elif cls.role == TensorRole.EMBEDDING:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        elif cls.role == TensorRole.OUTPUT_HEAD:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        else:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
    strategies["aggressive"] = MixedPrecisionPlan(
        model_identifier="mixed_aggressive",
        assignments=assignments,
    )
    
    # Strategy 3: Embedding + output head at INT8, rest INT4
    assignments = {}
    for name, cls in classifications.items():
        if cls.role in (TensorRole.EMBEDDING, TensorRole.OUTPUT_HEAD):
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        elif cls.role == TensorRole.NORM:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
        else:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT4,
                group_size=128,
            )
    strategies["int4_heavy"] = MixedPrecisionPlan(
        model_identifier="mixed_int4_heavy",
        assignments=assignments,
    )
    
    # Strategy 4: Binary for MLP, INT8 for attention
    assignments = {}
    for name, cls in classifications.items():
        if cls.role in (TensorRole.MLP_UP, TensorRole.MLP_DOWN, TensorRole.MLP_GATE):
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.BINARY,
                group_size=128,
            )
        elif cls.role in (TensorRole.ATTENTION_Q, TensorRole.ATTENTION_K,
                         TensorRole.ATTENTION_V, TensorRole.ATTENTION_O):
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        elif cls.role == TensorRole.NORM:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
        elif cls.role == TensorRole.EMBEDDING:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        elif cls.role == TensorRole.OUTPUT_HEAD:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.INT8,
                group_size=128,
            )
        else:
            assignments[name] = TensorPrecisionAssignment(
                tensor_name=name,
                precision=PrecisionType.FP16,
            )
    strategies["binary_mlp"] = MixedPrecisionPlan(
        model_identifier="mixed_binary_mlp",
        assignments=assignments,
    )
    
    return strategies


def run_scale_scenarios(
    scenarios: Optional[List[ModelScaleScenario]] = None,
    group_size: int = 128,
) -> List[ScenarioResult]:
    """Run mixed-precision analysis on model scale scenarios."""
    if scenarios is None:
        scenarios = get_default_scenarios()
    
    storage_model = MixedPrecisionStorageModel(group_size=group_size)
    results = []
    
    for scenario in scenarios:
        print(f"\n=== Analyzing {scenario.name} ({scenario.total_parameters/1e9:.1f}B params) ===")
        
        # Create classifications
        classifications = create_classifications_from_scenario(scenario)
        
        # Uniform FP16 baseline
        uniform_fp16_plan = create_uniform_fp16_plan(classifications)
        uniform_storage = storage_model.calculate_plan_storage(
            uniform_fp16_plan, classifications
        )
        
        uniform_results = {PrecisionType.FP16: uniform_storage}
        
        # Test other uniform precisions
        for prec in [PrecisionType.INT8, PrecisionType.INT4, PrecisionType.BINARY, PrecisionType.TERNARY]:
            plan = create_uniform_plan(f"uniform_{prec.value}", list(classifications.keys()), prec, group_size)
            storage = storage_model.calculate_plan_storage(plan, classifications)
            uniform_results[prec] = storage
        
        # Mixed precision strategies
        mixed_plans = create_mixed_precision_strategies(classifications)
        mixed_storage = {}
        storage_savings = {}
        
        fp16_bytes = uniform_storage.total_bytes
        
        for strat_name, plan in mixed_plans.items():
            storage = storage_model.calculate_plan_storage(plan, classifications)
            mixed_storage[strat_name] = storage
            savings = (fp16_bytes - storage.total_bytes) / fp16_bytes * 100
            storage_savings[strat_name] = savings
            print(f"  {strat_name}: {storage.total_bytes/1e9:.2f} GB ({savings:.1f}% vs FP16)")
        
        result = ScenarioResult(
            scenario=scenario,
            uniform_storage=uniform_results,
            mixed_storage=mixed_storage,
            mixed_plans=mixed_plans,
            storage_savings=storage_savings,
            parameter_distribution=param_dist,
        )
        results.append(result)
    
    return results


def analyze_scenario_results(results: List[ScenarioResult]) -> Dict[str, Any]:
    """Analyze and summarize scenario results."""
    summary = {
        "scenarios": [],
        "best_strategies": {},
    }
    
    for result in results:
        scenario_name = result.scenario.name
        param_dist = result.parameter_distribution
        
        # Find best strategy (highest savings with acceptable error)
        # Note: We don't have error estimates for scenarios without measurements
        best_strategy = max(result.storage_savings.items(), key=lambda x: x[1])
        
        summary["scenarios"].append({
            "name": scenario_name,
            "total_parameters": result.scenario.total_parameters,
            "uniform_fp16_gb": result.uniform_storage[PrecisionType.FP16].total_bytes / 1e9,
            "uniform_int4_gb": result.uniform_storage[PrecisionType.INT4].total_bytes / 1e9,
            "uniform_binary_gb": result.uniform_storage[PrecisionType.BINARY].total_bytes / 1e9,
            "mixed_strategies": {
                name: {
                    "storage_gb": storage.total_bytes / 1e9,
                    "savings_vs_fp16_pct": result.storage_savings[name],
                    "avg_bpw": storage.average_bits_per_weight,
                    "compression_ratio": storage.compression_ratio,
                }
                for name, storage in result.mixed_storage.items()
            },
            "best_savings_strategy": best_strategy[0],
            "best_savings_pct": best_strategy[1],
        })
    
    return summary