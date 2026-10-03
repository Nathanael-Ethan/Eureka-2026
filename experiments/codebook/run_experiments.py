"""
LDMARK Codebook Quantization Experiment Runner

Runs comprehensive codebook quantization experiments and exports results.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ldmark.compression.codebook import (
    CodebookExperimentConfig,
    run_codebook_experiment,
    run_uniform_vs_codebook_experiment,
    run_binary_codebook_experiment,
    generate_test_tensor,
    analyze_weight_distribution,
)
from ldmark.compression.codebook.export import (
    export_experiments_csv,
    export_experiments_json,
    export_comparison_csv,
)


def run_gaussian_experiments(output_dir: str = "codebook_results") -> None:
    """Run experiments on Gaussian-distributed tensors."""
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = [
        (128,),
        (1024,),
        (4096,),
        (256, 256),
        (1024, 1024),
    ]
    
    codebook_sizes = [2, 4, 8, 16, 32]
    group_sizes = [None, 128]
    
    print("\n=== Gaussian Tensor Experiments ===")
    all_results = []
    
    for shape in shapes:
        print(f"\nShape: {shape}")
        tensor = generate_test_tensor(shape, np.float32, "random_normal", seed=42)
        
        # Analyze distribution
        dist = analyze_weight_distribution(tensor)
        print(f"  Mean: {dist.mean:.4f}, Std: {dist.std:.4f}, Skew: {dist.skew:.4f}")
        
        for group_size in group_sizes:
            for cb_size in codebook_sizes:
                if group_size is None:
                    name = f"gaussian_{shape}_global_k{cb_size}"
                else:
                    name = f"gaussian_{shape}_group{group_size}_k{cb_size}"
                
                config = CodebookExperimentConfig(
                    experiment_name=name,
                    tensor_shape=shape,
                    source_dtype=np.float32,
                    codebook_size=cb_size,
                    group_size=group_size,
                    seed=42,
                    tensor_generator="custom",
                    custom_tensor=tensor,
                )
                
                result = run_codebook_experiment(config)
                all_results.append(result)
                
                cb_err = result.codebook_error
                unif_err = result.uniform_error
                print(f"  {name}: CB_MAE={cb_err['mae']:.6f}, U_MAE={unif_err['mae']:.6f}, "
                      f"CB_bpw={result.codebook_storage.effective_bits_per_weight:.3f}, "
                      f"U_bpw={result.uniform_storage['bits_per_weight']:.3f}")
    
    # Export
    export_experiments_csv(all_results, os.path.join(output_dir, "gaussian_experiments.csv"))
    export_experiments_json(all_results, os.path.join(output_dir, "gaussian_experiments.json"))
    print(f"\nResults exported to {output_dir}/")


def run_heavy_tail_experiments(output_dir: str = "codebook_results") -> None:
    """Run experiments on heavy-tailed tensors."""
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = [
        (1024,),
        (256, 256),
        (1024, 1024),
    ]
    
    codebook_sizes = [2, 4, 8, 16, 32]
    group_sizes = [None, 128]
    
    print("\n=== Heavy-Tailed Tensor Experiments ===")
    all_results = []
    
    for shape in shapes:
        print(f"\nShape: {shape}")
        tensor = generate_test_tensor(shape, np.float32, "heavy_tail", seed=42)
        
        dist = analyze_weight_distribution(tensor)
        print(f"  Mean: {dist.mean:.4f}, Std: {dist.std:.4f}, Skew: {dist.skew:.4f}, Kurt: {dist.kurtosis:.4f}")
        
        for group_size in group_sizes:
            for cb_size in codebook_sizes:
                if group_size is None:
                    name = f"heavy_tail_{shape}_global_k{cb_size}"
                else:
                    name = f"heavy_tail_{shape}_group{group_size}_k{cb_size}"
                
                config = CodebookExperimentConfig(
                    experiment_name=name,
                    tensor_shape=shape,
                    source_dtype=np.float32,
                    codebook_size=cb_size,
                    group_size=group_size,
                    seed=42,
                    tensor_generator="custom",
                    custom_tensor=tensor,
                )
                
                result = run_codebook_experiment(config)
                all_results.append(result)
                
                cb_err = result.codebook_error
                unif_err = result.uniform_error
                print(f"  {name}: CB_MAE={cb_err['mae']:.6f}, U_MAE={unif_err['mae']:.6f}")
    
    export_experiments_csv(all_results, os.path.join(output_dir, "heavy_tail_experiments.csv"))
    export_experiments_json(all_results, os.path.join(output_dir, "heavy_tail_experiments.json"))
    print(f"\nResults exported to {output_dir}/")


def run_transformer_like_experiments(output_dir: str = "codebook_results") -> None:
    """Run experiments on transformer-like synthetic tensors."""
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = [
        (512, 512),    # Attention weight
        (1024, 4096),  # MLP weight
        (4096, 1024),  # MLP output
    ]
    
    codebook_sizes = [2, 4, 8, 16, 32]
    group_sizes = [None, 128]
    
    print("\n=== Transformer-Like Tensor Experiments ===")
    all_results = []
    
    for shape in shapes:
        print(f"\nShape: {shape}")
        tensor = generate_test_tensor(shape, np.float32, "transformer_like", seed=42)
        
        dist = analyze_weight_distribution(tensor)
        print(f"  Mean: {dist.mean:.4f}, Std: {dist.std:.4f}, Skew: {dist.skew:.4f}")
        
        for group_size in group_sizes:
            for cb_size in codebook_sizes:
                if group_size is None:
                    name = f"transformer_{shape}_global_k{cb_size}"
                else:
                    name = f"transformer_{shape}_group{group_size}_k{cb_size}"
                
                config = CodebookExperimentConfig(
                    experiment_name=name,
                    tensor_shape=shape,
                    source_dtype=np.float32,
                    codebook_size=cb_size,
                    group_size=group_size,
                    seed=42,
                    tensor_generator="custom",
                    custom_tensor=tensor,
                )
                
                result = run_codebook_experiment(config)
                all_results.append(result)
                
                cb_err = result.codebook_error
                unif_err = result.uniform_error
                print(f"  {name}: CB_MAE={cb_err['mae']:.6f}, U_MAE={unif_err['mae']:.6f}")
    
    export_experiments_csv(all_results, os.path.join(output_dir, "transformer_experiments.csv"))
    export_experiments_json(all_results, os.path.join(output_dir, "transformer_experiments.json"))
    print(f"\nResults exported to {output_dir}/")


def run_binary_codebook_experiment(output_dir: str = "codebook_results") -> None:
    """Run binary vs learned 2-value codebook comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = [
        (128,),
        (1024,),
        (256, 256),
        (1024, 1024),
    ]
    
    generators = ["random_normal", "heavy_tail", "transformer_like"]
    
    print("\n=== Binary Codebook Experiments ===")
    all_results = []
    
    for shape in shapes:
        for gen in generators:
            print(f"\nShape: {shape}, Generator: {gen}")
            tensor = generate_test_tensor(shape, np.float32, gen, seed=42)
            
            result = run_binary_codebook_experiment(tensor, seed=42)
            result["shape"] = shape
            result["generator"] = gen
            all_results.append(result)
            
            fixed = result["fixed_binary"]
            learned = result["learned_binary"]
            imp = result["improvement"]
            
            print(f"  Fixed: MAE={fixed['mae']:.6f}, BPW={fixed['bpw']:.3f}")
            print(f"  Learned: MAE={learned['mae']:.6f}, BPW={learned['bpw']:.3f}")
            print(f"  MAE reduction: {imp['mae_reduction']*100:.1f}%")
    
    # Export binary experiment results
    import json
    with open(os.path.join(output_dir, "binary_codebook_experiments.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults exported to {output_dir}/")


def run_codebook_size_sweep(output_dir: str = "codebook_results") -> None:
    """Run comprehensive codebook size sweep."""
    os.makedirs(output_dir, exist_ok=True)
    
    tensor = generate_test_tensor((1024, 1024), np.float32, "transformer_like", seed=42)
    
    print("\n=== Codebook Size Sweep ===")
    
    sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    for cb_size in sizes:
        config = CodebookExperimentConfig(
            experiment_name=f"size_sweep_k{cb_size}",
            tensor_shape=tensor.shape,
            source_dtype=np.float32,
            codebook_size=cb_size,
            group_size=128,
            seed=42,
            tensor_generator="custom",
            custom_tensor=tensor,
        )
        
        result = run_codebook_experiment(config)
        results.append(result)
        
        cb_err = result.codebook_error
        unif_err = result.uniform_error
        cb_bpw = result.codebook_storage.effective_bits_per_weight
        
        print(f"  K={cb_size:3d}: MAE={cb_err['mae']:.6f}, U_MAE={unif_err['mae']:.6f}, "
              f"BPW={cb_bpw:.3f}, Ratio={cb_err['mae']/unif_err['mae']:.3f}")
    
    export_experiments_csv(results, os.path.join(output_dir, "codebook_size_sweep.csv"))
    export_experiments_json(results, os.path.join(output_dir, "codebook_size_sweep.json"))
    print(f"\nResults exported to {output_dir}/")


def main():
    """Main experiment runner."""
    print("=" * 60)
    print("LDMARK Codebook Quantization Laboratory - Experiment Runner")
    print("=" * 60)
    
    output_dir = "experiments/codebook/results"
    
    run_gaussian_experiments(output_dir)
    run_heavy_tail_experiments(output_dir)
    run_transformer_like_experiments(output_dir)
    run_binary_codebook_experiment(output_dir)
    run_codebook_size_sweep(output_dir)
    
    print("\n" + "=" * 60)
    print("ALL CODEBOOK EXPERIMENTS COMPLETE")
    print(f"Results available in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()