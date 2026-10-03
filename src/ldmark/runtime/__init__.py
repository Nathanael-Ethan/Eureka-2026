"""
LDMARK Runtime

Experimental runtime for loading LDMARK artifacts and reconstructing
compressed tensors for computation.

This is NOT a production inference engine. It's a correctness baseline
to prove the artifact -> runtime -> decompressed -> computation pipeline works.
"""

from .exceptions import (
    LDMARKRuntimeError,
    UnsupportedRuntimeFormat,
    TensorNotFoundError,
    ArtifactCorruptedError,
    DequantizationError,
    MemoryAccountingError,
    ComputationError,
)
from .memory import (
    MemoryAccountant,
    MemorySnapshot,
    estimate_decompressed_size,
    estimate_memory_impact,
    format_bytes,
    print_memory_comparison,
)
from .dequantize import (
    dequantize_tensor,
    DequantizedTensor,
    dequantize_int8,
    dequantize_int4,
    dequantize_binary,
    dequantize_ternary,
    validate_dequantization,
)
from .runtime import (
    LDMARKRuntime,
    LoadedTensor,
    MatmulResult,
    BenchmarkResult,
    open_runtime,
)
from .benchmark import (
    BenchmarkResult,
    AggregateBenchmark,
    run_full_benchmark,
    benchmark_single_tensor,
    print_benchmark_result,
)

__all__ = [
    # Exceptions
    "LDMARKRuntimeError",
    "UnsupportedRuntimeFormat",
    "TensorNotFoundError",
    "ArtifactCorruptedError",
    "DequantizationError",
    "MemoryAccountingError",
    "ComputationError",
    # Memory
    "MemoryAccountant",
    "MemorySnapshot",
    "estimate_decompressed_size",
    "estimate_memory_impact",
    "format_bytes",
    "print_memory_comparison",
    # Dequantization
    "dequantize_tensor",
    "DequantizedTensor",
    "dequantize_int8",
    "dequantize_int4",
    "dequantize_binary",
    "dequantize_ternary",
    "validate_dequantization",
    # Runtime
    "LDMARKRuntime",
    "LoadedTensor",
    "MatmulResult",
    "BenchmarkResult",
    "open_runtime",
    # Benchmark
    "BenchmarkResult",
    "AggregateBenchmark",
    "run_full_benchmark",
    "benchmark_single_tensor",
    "print_benchmark_result",
]

__version__ = "0.1.0"