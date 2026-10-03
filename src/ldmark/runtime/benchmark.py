"""
LDMARK Runtime - Benchmarking

Benchmarks for runtime operations: loading, reconstruction, computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import numpy as np

from .runtime import LDMARKRuntime, BenchmarkResult, open_runtime
from .memory import format_bytes


@dataclass
class AggregateBenchmark:
    """Aggregated benchmark results across multiple tensors."""
    tensor_results: List[BenchmarkResult]
    
    @property
    def total_tensors(self) -> int:
        return len(self.tensor_results)
    
    @property
    def avg_load_time_ms(self) -> float:
        return np.mean([r.load_time_ms for r in self.tensor_results]) if self.tensor_results else 0
    
    @property
    def avg_reconstruction_time_ms(self) -> float:
        return np.mean([r.reconstruction_time_ms for r in self.tensor_results]) if self.tensor_results else 0
    
    @property
    def avg_computation_time_ms(self) -> float:
        return np.mean([r.computation_time_ms for r in self.tensor_results]) if self.tensor_results else 0
    
    @property
    def total_compressed_bytes(self) -> int:
        return sum(r.compressed_bytes for r in self.tensor_results)
    
    @property
    def total_decompressed_bytes(self) -> int:
        return sum(r.decompressed_bytes for r in self.tensor_results)
    
    @property
    def avg_ops_per_second(self) -> float:
        return np.mean([r.ops_per_second for r in self.tensor_results]) if self.tensor_results else 0
    
    def to_dict(self) -> Dict:
        return {
            "total_tensors": self.total_tensors,
            "avg_load_time_ms": self.avg_load_time_ms,
            "avg_reconstruction_time_ms": self.avg_reconstruction_time_ms,
            "avg_computation_time_ms": self.avg_computation_time_ms,
            "total_compressed_bytes": self.total_compressed_bytes,
            "total_decompressed_bytes": self.total_decompressed_bytes,
            "expansion_factor": self.total_decompressed_bytes / self.total_compressed_bytes if self.total_compressed_bytes else 0,
            "avg_ops_per_second": self.avg_ops_per_second,
            "per_tensor": [r.to_dict() for r in self.tensor_results],
        }
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("=" * 70)
        print("LDMARK RUNTIME BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Tensors benchmarked:     {self.total_tensors}")
        print(f"Avg load time:           {self.avg_load_time_ms:.2f} ms")
        print(f"Avg reconstruction time: {self.avg_reconstruction_time_ms:.2f} ms")
        print(f"Avg computation time:    {self.avg_computation_time_ms:.2f} ms")
        print(f"Total compressed:        {format_bytes(self.total_compressed_bytes)}")
        print(f"Total decompressed:      {format_bytes(self.total_decompressed_bytes)}")
        if self.total_compressed_bytes > 0:
            print(f"Expansion factor:        {self.total_decompressed_bytes / self.total_compressed_bytes:.1f}x")
        print(f"Avg ops/second:          {self.avg_ops_per_second:,.0f}")
        print("=" * 70)


def run_full_benchmark(
    runtime: LDMARKRuntime,
    tensor_names: Optional[List[str]] = None,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> AggregateBenchmark:
    """
    Run benchmark on multiple tensors.
    
    Args:
        runtime: LDMARKRuntime instance
        tensor_names: List of tensor names (default: all tensors)
        input_shapes: Dict of tensor_name -> input_shape for matmul
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        
    Returns:
        AggregateBenchmark with all results
    """
    if tensor_names is None:
        tensor_names = runtime.list_tensors()
    
    results = []
    
    for name in tensor_names:
        # Determine input shape
        if input_shapes and name in input_shapes:
            shape = input_shapes[name]
        else:
            # Default: use weight shape to infer
            info = runtime.get_tensor_info(name)
            if info and len(info.original_shape) >= 2:
                in_features = info.original_shape[-1]
                shape = (1, in_features)  # batch=1
            else:
                shape = (1, 1024)  # fallback
        
        print(f"Benchmarking {name}...")
        result = runtime.benchmark_tensor(name, shape, num_warmup, num_runs)
        results.append(result)
        
        print(f"  Load: {result.load_time_ms:.2f}ms, "
              f"Reconstruct: {result.reconstruction_time_ms:.2f}ms, "
              f"Compute: {result.computation_time_ms:.2f}ms")
    
    return AggregateBenchmark(results)


def benchmark_single_tensor(
    artifact_dir: str,
    tensor_name: str,
    input_shape: Tuple[int, ...],
    num_warmup: int = 2,
    num_runs: int = 5,
    target_dtype: np.dtype = np.float16,
) -> BenchmarkResult:
    """
    Convenience function to benchmark a single tensor.
    
    Opens runtime, runs benchmark, closes runtime.
    """
    with open_runtime(artifact_dir, target_dtype=target_dtype) as runtime:
        return runtime.benchmark_tensor(tensor_name, input_shape, num_warmup, num_runs)


def print_benchmark_result(result: BenchmarkResult) -> None:
    """Print formatted benchmark result."""
    print("=" * 60)
    print(f"BENCHMARK: {result.tensor_name}")
    print("=" * 60)
    print(f"  Input shape:            {result.input_shape}")
    print(f"  Output shape:           {result.output_shape}")
    print(f"  Load time:              {result.load_time_ms:.2f} ms")
    print(f"  Reconstruction time:    {result.reconstruction_time_ms:.2f} ms")
    print(f"  Computation time:       {result.computation_time_ms:.2f} ms")
    print(f"  Total time:             {result.total_time_ms:.2f} ms")
    print(f"  Compressed size:        {format_bytes(result.compressed_bytes)}")
    print(f"  Decompressed size:      {format_bytes(result.decompressed_bytes)}")
    print(f"  Peak temp memory:       {format_bytes(result.peak_temporary_bytes)}")
    print(f"  Ops/second:             {result.ops_per_second:,.0f}")
    print("=" * 60)