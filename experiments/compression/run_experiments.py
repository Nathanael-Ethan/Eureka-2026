"""
Example experiment runner for LDMARK compression laboratory.

Run this to see the compression laboratory in action.
"""

import numpy as np
from src.ldmark.compression import (
    ExperimentConfig,
    QuantizationTarget,
    run_quantization_experiment,
    run_experiment_suite,
    calculate_prismml_storage,
    PrismMLConfig,
    validate_prismml_claim,
    compare_representations,
    prismml_q1_0_g128_bits_per_weight,
)


def main():
    print("=" * 60)
    print("LDMARK Compression Laboratory - Experiment Runner")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ExperimentConfig(
            experiment_name="fp32_int8_g128",
            input_shape=(512, 512),
            input_dtype=np.float32,
            target_bits=QuantizationTarget.INT8,
            group_size=128,
            seed=42,
        ),
        ExperimentConfig(
            experiment_name="fp32_int4_g128",
            input_shape=(512, 512),
            input_dtype=np.float32,
            target_bits=QuantizationTarget.INT4,
            group_size=128,
            seed=42,
        ),
        ExperimentConfig(
            experiment_name="fp16_int8_g128",
            input_shape=(512, 512),
            input_dtype=np.float16,
            target_bits=QuantizationTarget.INT8,
            group_size=128,
            seed=42,
        ),
        ExperimentConfig(
            experiment_name="fp16_int4_g128",
            input_shape=(512, 512),
            input_dtype=np.float16,
            target_bits=QuantizationTarget.INT4,
            group_size=128,
            seed=42,
        ),
    ]
    
    print("\n--- Running Quantization Experiments ---\n")
    results = run_experiment_suite(configs)
    
    for r in results:
        m = r.metrics
        print(f"Experiment: {r.config.experiment_name}")
        print(f"  ID: {r.experiment_id}")
        print(f"  Shape: {r.config.input_shape}, Dtype: {r.config.input_dtype}")
        print(f"  Target: {r.config.target_bits.name}, Group: {r.config.group_size}")
        print(f"  Bits/weight: {m.bits_per_weight:.4f}")
        print(f"  Compression ratio: {m.compression_ratio:.2f}x")
        print(f"  MAE: {m.mean_absolute_error:.6f}")
        print(f"  MSE: {m.mean_squared_error:.6f}")
        print(f"  Max Abs Error: {m.max_absolute_error:.6f}")
        print(f"  Relative Error: {m.relative_error:.6f}")
        print(f"  Original size: {m.original_size_bytes:,} bytes")
        print(f"  Compressed size: {m.compressed_size_bytes:,} bytes")
        print()
    
    # PrismML validation
    print("\n--- PrismML Q1_0_g128 Mathematics ---\n")
    
    # Calculate bits per weight
    bpw = prismml_q1_0_g128_bits_per_weight()
    print(f"PrismML Q1_0_g128 bits per weight: {bpw}")
    
    # Calculate storage for 27B model
    calc = calculate_prismml_storage(PrismMLConfig())
    print(f"Total bits: {calc['total_bits']:,.0f}")
    print(f"Total GB: {calc['total_gb']:.2f}")
    print(f"Group count: {calc['group_count']:,}")
    
    # Validate claimed 3.9 GB
    validation = validate_prismml_claim()
    print(f"\nClaimed: {validation['claimed_gb']} GB")
    print(f"Calculated: {validation['calculated_gb']:.2f} GB")
    print(f"Difference: {validation['difference_gb']:.2f} GB")
    print(f"Relative error: {validation['relative_error']*100:.2f}%")
    print(f"Within 10% tolerance: {validation['within_tolerance']}")
    
    # Compare representations
    print("\n--- Representation Comparison (27B params) ---\n")
    comparison = compare_representations()
    print(f"{'Representation':<25} {'Bits/Weight':>12} {'Total GB':>10}")
    print("-" * 50)
    for c in comparison['comparisons']:
        print(f"{c['representation']:<25} {c['bits_per_weight']:>12.4f} {c['total_gb']:>10.2f}")


if __name__ == "__main__":
    main()