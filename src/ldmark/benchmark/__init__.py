"""
LDMARK Benchmark Laboratory

Rigorous benchmarking subsystem for evaluating LDMARK compression methods.
Provides structured benchmark records, tensor-level benchmarking, method comparison,
storage validation, export functionality, repeatability verification, and regression benchmarks.
"""

from .model import (
    BenchmarkRecord,
    BenchmarkConfig,
    HardwareMetadata,
    TensorBenchmarkResult,
    ModelBenchmarkResult,
    ErrorDistribution,
    TimingMetrics,
    StorageMetrics,
)
from .runner import (
    run_tensor_benchmark,
    run_model_benchmark,
    run_method_comparison,
    run_repeatability_benchmark,
    create_synthetic_model,
)
from .storage import (
    validate_storage,
    calculate_theoretical_storage,
    calculate_actual_storage,
    StorageValidationResult,
)
from .export import (
    export_benchmarks_csv,
    export_benchmarks_json,
    export_tensor_results_csv,
    export_model_results_csv,
    export_error_distributions_csv,
    export_comparison_table_csv,
    load_benchmarks_json,
    load_benchmarks_csv,
)
from .regression import (
    RegressionThresholds,
    run_regression_benchmarks,
    RegressionResult,
    create_regression_suite,
    load_baseline,
)
from .prismml import (
    run_prismml_q1_0_g128_benchmark,
    PrismMLBenchmarkResult,
    run_prismml_scaling_benchmark,
    run_prismml_claim_validation,
    create_prismml_comparison_table,
    export_prismml_benchmark_report,
    run_full_prismml_validation_suite,
)

__all__ = [
    # Data model
    "BenchmarkRecord",
    "BenchmarkConfig",
    "HardwareMetadata",
    "TensorBenchmarkResult",
    "ModelBenchmarkResult",
    "ErrorDistribution",
    "TimingMetrics",
    "StorageMetrics",
    # Runner
    "run_tensor_benchmark",
    "run_model_benchmark",
    "run_method_comparison",
    "run_repeatability_benchmark",
    "create_synthetic_model",
    # Storage validation
    "validate_storage",
    "calculate_theoretical_storage",
    "calculate_actual_storage",
    "StorageValidationResult",
    # Export
    "export_benchmarks_csv",
    "export_benchmarks_json",
    "export_tensor_results_csv",
    "export_model_results_csv",
    "export_error_distributions_csv",
    "export_comparison_table_csv",
    "load_benchmarks_json",
    "load_benchmarks_csv",
    # Regression
    "RegressionThresholds",
    "run_regression_benchmarks",
    "RegressionResult",
    "create_regression_suite",
    "load_baseline",
    # PrismML
    "run_prismml_q1_0_g128_benchmark",
    "PrismMLBenchmarkResult",
    "run_prismml_scaling_benchmark",
    "run_prismml_claim_validation",
    "create_prismml_comparison_table",
    "export_prismml_benchmark_report",
    "run_full_prismml_validation_suite",
]