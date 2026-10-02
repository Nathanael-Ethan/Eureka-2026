"""
LDMARK Benchmark Laboratory - Experiment Runner

Runs comprehensive compression benchmarks and exports results for analysis.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ldmark.benchmark import (
    BenchmarkConfig,
    run_tensor_benchmark,
    run_model_benchmark,
    run_method_comparison,
    run_repeatability_benchmark,
    run_prismml_q1_0_g128_benchmark,
    run_full_prismml_validation_suite,
    create_synthetic_model,
    export_benchmarks_csv,
    export_benchmarks_json,
    export_tensor_results_csv,
    export_model_results_csv,
    export_error_distributions_csv,
    export_comparison_table_csv,
    RegressionThresholds,
    run_regression_benchmarks,
    create_regression_suite,
)
from ldmark.benchmark.model import BenchmarkRecord, HardwareMetadata
from ldmark.compression.quantize import QuantizationTarget


def run_tensor_level_benchmarks(output_dir: str = "benchmark_results") -> None:
    """Run tensor-level benchmarks across methods and configurations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Test tensor configurations
    tensor_configs = [
        ("small_vector", (128,), np.float32),
        ("medium_vector", (1024,), np.float32),
        ("large_vector", (8192,), np.float32),
        ("small_matrix", (64, 64), np.float32),
        ("medium_matrix", (256, 256), np.float32),
        ("large_matrix", (1024, 1024), np.float32),
        ("fp16_vector", (1024,), np.float16),
        ("fp16_matrix", (256, 256), np.float16),
    ]
    
    # Method configurations
    methods = [
        ("FP32", 32, 0, np.float32),
        ("FP16", 16, 0, np.float16),
        ("INT8", 8, 128, np.float16),
        ("INT4", 4, 128, np.float16),
        ("BINARY", 1, 128, np.float16),
        ("TERNARY", int(np.log2(3) * 1000) / 1000, 128, np.float16),
    ]
    
    all_records = []
    all_tensor_results = []
    
    for name, shape, dtype in tensor_configs:
        print(f"\nBenchmarking tensor: {name} {shape} {dtype}")
        tensor = np.random.default_rng(42).normal(0, 1, shape).astype(dtype)
        
        for method, target_bits, group_size, scale_dtype in methods:
            if method in ("FP32", "FP16") and group_size > 0:
                continue  # Skip grouped for non-grouped methods
            
            try:
                result = run_tensor_benchmark(
                    tensor=tensor,
                    tensor_name=name,
                    compression_method=method,
                    target_bits=target_bits,
                    group_size=group_size if group_size > 0 else 1,
                    scale_dtype=scale_dtype,
                    random_seed=42,
                )
                all_tensor_results.append(result)
                
                # Create full BenchmarkRecord
                record = BenchmarkRecord(
                    experiment_id=f"{name}_{method}_g{group_size}",
                    model_identifier=name,
                    parameter_count=tensor.size,
                    tensor_count=1,
                    source_dtype=str(dtype),
                    compression_method=method,
                    target_bits=target_bits,
                    group_size=group_size if group_size > 0 else 1,
                    scale_dtype=str(scale_dtype),
                    original_bytes=result.original_bytes,
                    compressed_bytes=result.compressed_bytes,
                    actual_bits_per_weight=result.actual_bits_per_weight,
                    compression_ratio=result.compression_ratio,
                    mae=result.error_distribution.mean,
                    mse=result.error_distribution.mean ** 2,
                    rmse=result.error_distribution.mean,
                    max_absolute_error=result.error_distribution.maximum,
                    relative_error=result.relative_error,
                    error_distribution=result.error_distribution,
                    transformation_time_ms=result.timing.transformation_time_ms,
                    dequantization_time_ms=result.timing.dequantization_time_ms,
                    validation_time_ms=result.timing.validation_time_ms,
                    total_time_ms=result.timing.total_time_ms,
                    theoretical_storage_bytes=result.storage.theoretical_bytes,
                    actual_storage_bytes=result.storage.actual_bytes,
                    metadata_bytes=result.storage.metadata_bytes,
                    scales_bytes=result.storage.scales_bytes,
                    padding_bytes=result.storage.padding_bytes,
                    tensor_headers_bytes=result.storage.tensor_headers_bytes,
                    serialization_overhead_bytes=result.storage.serialization_overhead_bytes,
                    hardware_metadata=HardwareMetadata.capture(),
                    random_seed=42,
                    timestamp="",
                )
                all_records.append(record)
                
                print(f"  {method}: {result.actual_bits_per_weight:.4f} bpw, "
                      f"{result.compression_ratio:.2f}x, "
                      f"MAE={result.error_distribution.mean:.6f}")
            except Exception as e:
                print(f"  {method}: FAILED - {e}")
    
    # Export results
    export_benchmarks_csv(all_records, os.path.join(output_dir, "tensor_benchmarks.csv"))
    export_benchmarks_json(all_records, os.path.join(output_dir, "tensor_benchmarks.json"))
    export_tensor_results_csv(all_tensor_results, os.path.join(output_dir, "tensor_results.csv"))
    export_error_distributions_csv(all_records, os.path.join(output_dir, "error_distributions.csv"))
    export_comparison_table_csv(all_records, os.path.join(output_dir, "comparison_table.csv"))
    
    print(f"\nTensor benchmarks complete. Results in {output_dir}/")
    print(f"  Total records: {len(all_records)}")


def run_model_level_benchmarks(output_dir: str = "benchmark_results") -> None:
    """Run model-level benchmarks on synthetic models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Synthetic model configurations
    model_configs = [
        ("tiny_model", [
            ("embedding", (1000, 64), "float32"),
            ("layer1_weight", (64, 128), "float32"),
            ("layer1_bias", (128,), "float32"),
            ("layer2_weight", (128, 10), "float32"),
            ("layer2_bias", (10,), "float32"),
        ]),
        ("small_model", [
            ("embedding", (10000, 256), "float32"),
            ("encoder_layers.0.weight", (256, 512), "float32"),
            ("encoder_layers.0.bias", (512,), "float32"),
            ("encoder_layers.1.weight", (512, 256), "float32"),
            ("encoder_layers.1.bias", (256,), "float32"),
            ("decoder_weight", (256, 10000), "float32"),
        ]),
        ("medium_model_fp16", [
            ("embedding", (50000, 512), "float16"),
            ("layers.0.attn.qkv.weight", (512, 1536), "float16"),
            ("layers.0.attn.out.weight", (512, 512), "float16"),
            ("layers.0.mlp.weight", (512, 2048), "float16"),
            ("layers.0.mlp.out.weight", (2048, 512), "float16"),
            ("layers.1.attn.qkv.weight", (512, 1536), "float16"),
            ("layers.1.attn.out.weight", (512, 512), "float16"),
            ("layers.1.mlp.weight", (512, 2048), "float16"),
            ("layers.1.mlp.out.weight", (2048, 512), "float16"),
            ("output_weight", (512, 50000), "float16"),
        ]),
    ]
    
    methods = [
        ("FP32", 32, 1, np.float32),
        ("FP16", 16, 1, np.float16),
        ("INT8", 8, 128, np.float16),
        ("INT4", 4, 128, np.float16),
    ]
    
    all_model_results = []
    all_records = []
    
    for model_name, tensor_specs in model_configs:
        print(f"\nBenchmarking model: {model_name}")
        tensors = create_synthetic_model(tensor_specs, generator="random_normal", seed=42)
        
        for method, target_bits, group_size, scale_dtype in methods:
            if method in ("FP32", "FP16") and target_bits != (32 if method == "FP32" else 16):
                continue
            
            try:
                result = run_model_benchmark(
                    tensors=tensors,
                    model_identifier=model_name,
                    compression_method=method,
                    target_bits=target_bits,
                    group_size=group_size,
                    scale_dtype=scale_dtype,
                    random_seed=42,
                )
                all_model_results.append(result)
                
                # Create BenchmarkRecord for model aggregate
                record = BenchmarkRecord(
                    experiment_id=f"{model_name}_{method}_g{group_size}",
                    model_identifier=model_name,
                    parameter_count=result.parameter_count,
                    tensor_count=result.tensor_count,
                    source_dtype="mixed" if method in ("FP32", "FP16") else str(scale_dtype),
                    compression_method=method,
                    target_bits=target_bits,
                    group_size=group_size,
                    scale_dtype=str(scale_dtype),
                    original_bytes=result.aggregate_original_bytes,
                    compressed_bytes=result.aggregate_compressed_bytes,
                    actual_bits_per_weight=result.aggregate_bits_per_weight,
                    compression_ratio=result.aggregate_compression_ratio,
                    mae=0.0,  # Aggregate - would need per-tensor averaging
                    mse=0.0,
                    rmse=0.0,
                    max_absolute_error=0.0,
                    relative_error=0.0,
                    error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,0),
                    transformation_time_ms=result.total_timing.transformation_time_ms,
                    dequantization_time_ms=result.total_timing.dequantization_time_ms,
                    validation_time_ms=result.total_timing.validation_time_ms,
                    total_time_ms=result.total_timing.total_time_ms,
                    theoretical_storage_bytes=result.aggregate_storage.theoretical_bytes,
                    actual_storage_bytes=result.aggregate_storage.actual_bytes,
                    metadata_bytes=result.aggregate_storage.metadata_bytes,
                    scales_bytes=result.aggregate_storage.scales_bytes,
                    padding_bytes=result.aggregate_storage.padding_bytes,
                    tensor_headers_bytes=result.aggregate_storage.tensor_headers_bytes,
                    serialization_overhead_bytes=result.aggregate_storage.serialization_overhead_bytes,
                    hardware_metadata=HardwareMetadata.capture(),
                    random_seed=42,
                    timestamp="",
                )
                all_records.append(record)
                
                print(f"  {method}: {result.aggregate_bits_per_weight:.4f} bpw, "
                      f"{result.aggregate_compression_ratio:.2f}x, "
                      f"{result.parameter_count:,} params")
            except Exception as e:
                print(f"  {method}: FAILED - {e}")
    
    # Export
    export_benchmarks_csv(all_records, os.path.join(output_dir, "model_benchmarks.csv"))
    export_benchmarks_json(all_records, os.path.join(output_dir, "model_benchmarks.json"))
    export_model_results_csv(all_model_results, os.path.join(output_dir, "model_results.csv"))
    
    print(f"\nModel benchmarks complete. Results in {output_dir}/")
    print(f"  Total model records: {len(all_records)}")


def run_repeatability_experiments(output_dir: str = "benchmark_results") -> None:
    """Run repeatability verification experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    tensor = np.random.default_rng(42).normal(0, 1, (1024, 1024)).astype(np.float32)
    
    methods = [
        ("INT8", 8, 128, np.float16),
        ("INT4", 4, 128, np.float16),
        ("BINARY", 1, 128, np.float16),
    ]
    
    print("\nRunning repeatability experiments...")
    for method, target_bits, group_size, scale_dtype in methods:
        print(f"  {method}...")
        results, analysis = run_repeatability_benchmark(
            tensor=tensor,
            tensor_name="repeatability_test",
            compression_method=method,
            target_bits=target_bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            num_runs=10,
            base_seed=42,
        )
        
        print(f"    Identical results: {analysis['all_results_identical']}")
        print(f"    BPW: mean={analysis['bits_per_weight']['mean']:.6f}, "
              f"std={analysis['bits_per_weight']['std']:.6f}")
        print(f"    Timing: mean={analysis['timing_ms']['mean']:.2f}ms, "
              f"std={analysis['timing_ms']['std']:.2f}ms")
    
    # Save analysis
    import json
    with open(os.path.join(output_dir, "repeatability_analysis.json"), 'w') as f:
        json.dump(analysis, f, indent=2)


def run_prismml_experiments(output_dir: str = "benchmark_results") -> None:
    """Run PrismML Q1_0_g128 validation experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nRunning PrismML Q1_0_g128 validation...")
    
    # Full validation suite
    suite_results = run_full_prismml_validation_suite(
        tensor_shapes=[
            (128,), (256,), (1024,), (4096,),
            (64, 64), (256, 256), (1024, 1024),
        ],
        output_dir=os.path.join(output_dir, "prismml"),
    )
    
    print(f"  Mathematical BPW: {suite_results['mathematical_validation']['bpw']}")
    print(f"  Scaling constant: {suite_results['scaling_validation']['bpw_constant_across_sizes']}")
    print(f"  Claim validation: {suite_results['claim_validation']['calculated_gb']:.2f} GB calculated vs "
          f"{suite_results['claim_validation']['claimed_gb']} GB claimed")
    print(f"  Tensor benchmarks: {len(suite_results['tensor_benchmarks'])} shapes tested")


def run_regression_suite_experiment(output_dir: str = "benchmark_results") -> None:
    """Create and run regression benchmark suite."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating regression baseline suite...")
    
    tensor_specs = [
        ("test_vector", (1024,), "float32"),
        ("test_matrix", (256, 256), "float32"),
        ("test_fp16", (512, 512), "float16"),
    ]
    
    methods = [
        {"compression_method": "INT8", "target_bits": 8, "group_size": 128, "scale_dtype": "float16"},
        {"compression_method": "INT4", "target_bits": 4, "group_size": 128, "scale_dtype": "float16"},
        {"compression_method": "BINARY", "target_bits": 1, "group_size": 128, "scale_dtype": "float16"},
    ]
    
    baseline = create_regression_suite(
        tensor_specs=tensor_specs,
        methods=methods,
        seeds=[42, 123],
        output_dir=os.path.join(output_dir, "regression_baselines"),
    )
    
    print(f"  Created {len(baseline)} baseline records")
    
    # Run regression check against same tensors (should pass)
    current_tensors = create_synthetic_model(tensor_specs, seed=42)
    
    result = run_regression_benchmarks(
        baseline_records=baseline,
        current_tensors=current_tensors,
        config={},
    )
    
    print(f"  Regression check: {'PASSED' if result.overall_passed else 'FAILED'}")
    print(f"  Checks: {result.summary['passed']}/{result.summary['total_checks']} passed")
    if result.summary['regressions_detected']:
        print(f"  Regressions: {result.summary['regression_details']}")
    
    # Export
    with open(os.path.join(output_dir, "regression_result.json"), 'w') as f:
        f.write(result.to_json())


def main():
    """Main experiment runner."""
    print("=" * 60)
    print("LDMARK Benchmark Laboratory - Experiment Runner")
    print("=" * 60)
    
    output_dir = "experiments/benchmarks/results"
    
    # Run all benchmark categories
    run_tensor_level_benchmarks(output_dir)
    run_model_level_benchmarks(output_dir)
    run_repeatability_experiments(output_dir)
    run_prismml_experiments(output_dir)
    run_regression_suite_experiment(output_dir)
    
    print("\n" + "=" * 60)
    print("ALL BENCHMARKS COMPLETE")
    print(f"Results available in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()