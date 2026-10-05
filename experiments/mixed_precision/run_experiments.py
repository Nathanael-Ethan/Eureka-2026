"""
LDMARK Mixed-Precision Experiment Runner

Runs experiments demonstrating mixed-precision vs uniform precision strategies.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ldmark.mixed_precision import (
    PrecisionType,
    TensorRole,
    TensorClassification,
    TensorPrecisionAssignment,
    MixedPrecisionPlan,
    create_mixed_precision_plan,
    create_uniform_plan,
    classify_tensor,
    TensorClassifier,
    MixedPrecisionStorageModel,
    calculate_mixed_precision_storage,
    run_sensitivity_experiment,
    measure_tensor_errors,
    SearchConstraint,
    greedy_search,
    get_default_scenarios,
    estimate_parameter_distribution,
    create_classifications_from_scenario,
    create_mixed_precision_strategies,
    run_scale_scenarios,
    export_plan_csv,
    export_plan_json,
    export_storage_accounting_csv,
    export_scenario_results_csv,
    export_search_results_csv,
    export_error_profiles_csv,
)


def run_sensitivity_experiment_on_synthetic(output_dir: str = "mixed_precision_results") -> None:
    """Run sensitivity experiment on synthetic transformer tensors."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Sensitivity Experiment ===")
    
    # Create synthetic transformer-like tensors
    tensors = {
        "embed_tokens.weight": np.random.randn(32000, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.q_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.k_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.v_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.o_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.mlp.up_proj.weight": np.random.randn(4096, 11008).astype(np.float32) * 0.02,
        "layers.0.mlp.down_proj.weight": np.random.randn(11008, 4096).astype(np.float32) * 0.02,
        "layers.0.norm.weight": np.ones(4096).astype(np.float32),
        "lm_head.weight": np.random.randn(4096, 32000).astype(np.float32) * 0.02,
    }
    
    # Classify tensors
    classifier = TensorClassifier(architecture="llama")
    classifications = classifier.classify_model(tensors)
    
    print(f"Classified {len(classifications)} tensors")
    role_dist = classifier.get_role_distribution(classifications)
    for role, count in role_dist.items():
        print(f"  {role.value}: {count} tensors")
    
    # Run sensitivity experiment
    print("\nRunning sensitivity experiment...")
    error_profiles = run_sensitivity_experiment(
        tensors,
        classifications,
        group_size=128,
    )
    
    # Export error profiles
    export_error_profiles_csv(error_profiles, os.path.join(output_dir, "error_profiles.csv"))
    print(f"Error profiles exported to {output_dir}/error_profiles.csv")
    
    # Print summary
    print("\nError Summary (MAE):")
    for name, profile in error_profiles.items():
        role = profile.tensor_classification.role.value
        fp16 = profile.measurements.get(PrecisionType.FP16)
        int8 = profile.measurements.get(PrecisionType.INT8)
        int4 = profile.measurements.get(PrecisionType.INT4)
        
        print(f"  {name} ({role}):")
        if fp16: print(f"    FP16: MAE={fp16.mae:.6f}")
        if int8: print(f"    INT8: MAE={int8.mae:.6f}, BPW={int8.effective_bits_per_weight:.2f}")
        if int4: print(f"    INT4: MAE={int4.mae:.6f}, BPW={int4.effective_bits_per_weight:.2f}")


def run_storage_comparison(output_dir: str = "mixed_precision_results") -> None:
    """Compare storage of uniform vs mixed precision."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Storage Comparison ===")
    
    # Create synthetic model
    tensors = {
        "embed_tokens.weight": np.random.randn(32000, 4096).astype(np.float32),
        "layers.0.attn.q_proj.weight": np.random.randn(4096, 4096).astype(np.float32),
        "layers.0.attn.k_proj.weight": np.random.randn(4096, 4096).astype(np.float32),
        "layers.0.attn.v_proj.weight": np.random.randn(4096, 4096).astype(np.float32),
        "layers.0.attn.o_proj.weight": np.random.randn(4096, 4096).astype(np.float32),
        "layers.0.mlp.up_proj.weight": np.random.randn(4096, 11008).astype(np.float32),
        "layers.0.mlp.down_proj.weight": np.random.randn(11008, 4096).astype(np.float32),
        "layers.0.norm.weight": np.ones(4096).astype(np.float32),
        "lm_head.weight": np.random.randn(4096, 32000).astype(np.float32),
    }
    
    classifier = TensorClassifier(architecture="llama")
    classifications = classifier.classify_model(tensors)
    
    storage_model = MixedPrecisionStorageModel(group_size=128)
    
    # Uniform precision plans
    tensor_names = list(classifications.keys())
    
    print("Uniform Precision Plans:")
    uniform_plans = {}
    for prec in [PrecisionType.FP16, PrecisionType.INT8, PrecisionType.INT4, PrecisionType.BINARY]:
        plan = create_uniform_plan(f"uniform_{prec.value}", tensor_names, prec, group_size=128)
        storage = storage_model.calculate_plan_storage(plan, classifications)
        uniform_plans[prec] = storage
        print(f"  {prec.value}: {storage.total_bytes/1e9:.3f} GB, {storage.average_bits_per_weight:.2f} bpw, {storage.compression_ratio:.1f}x")
    
    # Mixed precision strategies
    print("\nMixed Precision Strategies:")
    mixed_strategies = create_mixed_precision_strategies(classifications)
    
    fp16_bytes = uniform_plans[PrecisionType.FP16].total_bytes
    
    for name, plan in mixed_strategies.items():
        storage = storage_model.calculate_plan_storage(plan, classifications)
        savings = (fp16_bytes - storage.total_bytes) / fp16_bytes * 100
        print(f"  {name}: {storage.total_bytes/1e9:.3f} GB ({savings:.1f}% vs FP16), {storage.average_bits_per_weight:.2f} bpw")
    
    # Export storage accounting
    for prec, storage in uniform_plans.items():
        export_storage_accounting_csv(storage, os.path.join(output_dir, f"uniform_{prec.value}_storage.csv"))
    
    for name, plan in mixed_strategies.items():
        storage = storage_model.calculate_plan_storage(plan, classifications)
        export_storage_accounting_csv(storage, os.path.join(output_dir, f"mixed_{name}_storage.csv"))
        export_plan_csv(plan, os.path.join(output_dir, f"plan_{name}.csv"))
        export_plan_json(plan, os.path.join(output_dir, f"plan_{name}.json"))
    
    print(f"\nResults exported to {output_dir}/")


def run_constraint_search(output_dir: str = "mixed_precision_results") -> None:
    """Run constraint-based mixed-precision search."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Constraint Search ===")
    
    # Create synthetic model with error profiles
    tensors = {
        "embed_tokens.weight": np.random.randn(32000, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.q_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.k_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.v_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.attn.o_proj.weight": np.random.randn(4096, 4096).astype(np.float32) * 0.02,
        "layers.0.mlp.up_proj.weight": np.random.randn(4096, 11008).astype(np.float32) * 0.02,
        "layers.0.mlp.down_proj.weight": np.random.randn(11008, 4096).astype(np.float32) * 0.02,
        "layers.0.norm.weight": np.ones(4096).astype(np.float32),
        "lm_head.weight": np.random.randn(4096, 32000).astype(np.float32) * 0.02,
    }
    
    classifier = TensorClassifier(architecture="llama")
    classifications = classifier.classify_model(tensors)
    
    # Get error profiles
    print("Generating error profiles...")
    error_profiles = run_sensitivity_experiment(
        tensors,
        classifications,
        group_size=128,
    )
    
    # Define constraints
    constraints = SearchConstraint(
        max_storage_gb=1.5,  # 1.5 GB limit
        max_average_bits_per_weight=6.0,
        max_mae=0.1,
    )
    
    print(f"\nConstraints: {constraints.to_dict()}")
    
    # Run greedy search
    print("\nRunning greedy search...")
    result = greedy_search(
        classifications=classifications,
        error_profiles=error_profiles,
        storage_model=MixedPrecisionStorageModel(group_size=128),
        constraints=constraints,
    )
    
    print(f"\nFeasible configurations: {len(result.feasible_configurations)}")
    print(f"Infeasible configurations: {len(result.infeasible_configurations)}")
    
    if result.feasible_configurations:
        print("\nFeasible configurations:")
        for i, config in enumerate(result.feasible_configurations):
            print(f"  {i+1}: {config.storage.total_bytes/1e9:.3f} GB, "
                  f"{config.storage.average_bits_per_weight:.2f} bpw, "
                  f"MAE={config.error_metrics.get('mae', 0):.6f}, "
                  f"Precisions: {config.plan.to_dict()['precision_distribution']}")
    
    # Export results
    export_search_results_csv(result, os.path.join(output_dir, "search_results.csv"))
    export_search_results_json(result, os.path.join(output_dir, "search_results.json"))
    print(f"\nSearch results exported to {output_dir}/")


def run_model_scale_scenarios(output_dir: str = "mixed_precision_results") -> None:
    """Run model-scale scenarios (7B, 13B, 27B, 70B)."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== Model Scale Scenarios ===")
    
    results = run_scale_scenarios()
    
    for result in results:
        print(f"\n{result.scenario.name} ({result.scenario.total_parameters/1e9:.1f}B params):")
        
        # Uniform storage
        for prec, storage in result.uniform_storage.items():
            print(f"  Uniform {prec.value}: {storage.total_bytes/1e9:.3f} GB, "
                  f"{storage.average_bits_per_weight:.2f} bpw, {storage.compression_ratio:.1f}x")
        
        # Mixed storage
        for name, storage in result.mixed_storage.items():
            savings = result.storage_savings.get(name, 0)
            print(f"  Mixed {name}: {storage.total_bytes/1e9:.3f} GB "
                  f"({savings:.1f}% vs FP16), {storage.average_bits_per_weight:.2f} bpw")
    
    # Export scenario results
    for result in results:
        export_scenario_results_csv(result, os.path.join(output_dir, f"scenario_{result.scenario.name}.csv"))
        export_scenario_results_json(result, os.path.join(output_dir, f"scenario_{result.scenario.name}.json"))
    
    print(f"\nScenario results exported to {output_dir}/")


def main():
    """Main experiment runner."""
    print("=" * 60)
    print("LDMARK Mixed-Precision Experiment Runner")
    print("=" * 60)
    
    output_dir = "experiments/mixed_precision/results"
    
    run_sensitivity_experiment_on_synthetic(output_dir)
    run_storage_comparison(output_dir)
    run_constraint_search(output_dir)
    run_model_scale_scenarios(output_dir)
    
    print("\n" + "=" * 60)
    print("ALL MIXED-PRECISION EXPERIMENTS COMPLETE")
    print(f"Results available in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()