"""
LDMARK Runtime - Main Runtime Interface

Lightweight experimental runtime for loading LDMARK artifacts
and reconstructing compressed tensors for computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np

from .exceptions import (
    LDMARKRuntimeError,
    UnsupportedRuntimeFormat,
    TensorNotFoundError,
    ArtifactCorruptedError,
    DequantizationError,
    ComputationError,
)
from .memory import MemoryAccountant, MemorySnapshot
from .dequantize import dequantize_tensor, DequantizedTensor, validate_dequantization
from src.ldmark.artifact.reader import LDMARKArtifactReader, open_artifact, TensorData
from src.ldmark.artifact.format import (
    Manifest,
    TensorIndexEntry,
    TensorEncoding,
    ArtifactFormatVersion,
    CompressionMethod,
    ModelInfo,
    CompressionInfo,
)
from src.ldmark.compression.metrics import CompressionMetrics, calculate_error_metrics


@dataclass
class LoadedTensor:
    """A tensor that has been loaded and optionally decompressed."""
    name: str
    compressed_data: np.ndarray
    scales: Optional[np.ndarray]
    entry: TensorIndexEntry
    decompressed_data: Optional[np.ndarray] = None
    dequantized: Optional[DequantizedTensor] = None
    is_decompressed: bool = False
    
    def decompress(self, target_dtype: np.dtype = np.float16) -> DequantizedTensor:
        """Decompress the tensor if not already done."""
        if not self.is_decompressed:
            self.dequantized = dequantize_tensor(
                self.compressed_data,
                self.scales,
                self.entry,
                target_dtype
            )
            self.decompressed_data = self.dequantized.data
            self.is_decompressed = True
        return self.dequantized
    
    def get_data(self, target_dtype: np.dtype = np.float16) -> np.ndarray:
        """Get tensor data, decompressing if necessary."""
        if not self.is_decompressed:
            self.decompress(target_dtype)
        return self.decompressed_data
    
    def release_decompressed(self) -> None:
        """Release decompressed data to free memory."""
        self.decompressed_data = None
        self.dequantized = None
        self.is_decompressed = False


@dataclass
class MatmulResult:
    """Result of matrix multiplication."""
    output: np.ndarray
    weight_name: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    computation_time_ms: float
    memory_snapshot: MemorySnapshot


@dataclass
class BenchmarkResult:
    """Result of a runtime benchmark."""
    tensor_name: str
    load_time_ms: float
    reconstruction_time_ms: float
    computation_time_ms: float
    total_time_ms: float
    compressed_bytes: int
    decompressed_bytes: int
    peak_temporary_bytes: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    ops_per_second: float


class LDMARKRuntime:
    """
    Main runtime interface for LDMARK artifacts.
    
    Features:
    - Lazy tensor loading (load only what you need)
    - Format-aware dequantization (INT8, INT4, binary, ternary)
    - Memory accounting (compressed vs decompressed vs temporary)
    - Basic numpy computation (matmul)
    - Benchmarking support
    
    Example:
        runtime = LDMARKRuntime.open("path/to/artifact")
        runtime.get_tensor("layer.0.weight")  # Loads only this tensor
        output = runtime.matmul("layer.0.weight", input_array)
        runtime.close()
    """
    
    def __init__(
        self,
        artifact_dir: Union[str, Path],
        verify_checksums: bool = True,
        mmap_tensors: bool = False,
        target_dtype: np.dtype = np.float16,
    ):
        """
        Initialize runtime with an LDMARK artifact.
        
        Args:
            artifact_dir: Path to artifact directory
            verify_checksums: Whether to verify SHA-256 checksums
            mmap_tensors: Whether to memory-map tensor files
            target_dtype: Default dtype for dequantized tensors
        """
        self.artifact_dir = Path(artifact_dir).resolve()
        self.target_dtype = target_dtype
        self._reader: Optional[LDMARKArtifactReader] = None
        self._loaded_tensors: Dict[str, LoadedTensor] = {}
        self._memory = MemoryAccountant()
        self._is_open = False
        self._verify_checksums = verify_checksums
        self._mmap_tensors = mmap_tensors
        
        # Open the artifact
        self.open()
    
    @classmethod
    def open(
        cls,
        artifact_dir: Union[str, Path],
        verify_checksums: bool = True,
        mmap_tensors: bool = False,
        target_dtype: np.dtype = np.float16,
    ) -> "LDMARKRuntime":
        """
        Open an LDMARK artifact and return a runtime instance.
        
        This is the primary entry point.
        """
        return cls(artifact_dir, verify_checksums, mmap_tensors, target_dtype)
    
    def open(self) -> None:
        """Open the artifact (called automatically on init)."""
        if self._is_open:
            return
        
        try:
            self._reader = open_artifact(
                self.artifact_dir,
                verify_checksums=self._verify_checksums,
                mmap_tensors=self._mmap_tensors,
            )
            self._is_open = True
        except Exception as e:
            raise ArtifactCorruptedError(f"Failed to open artifact: {e}") from e
    
    def close(self) -> None:
        """Close the artifact and release all resources."""
        # Release all decompressed tensors
        for tensor in self._loaded_tensors.values():
            tensor.release_decompressed()
        self._loaded_tensors.clear()
        self._memory.reset()
        
        if self._reader:
            self._reader.__exit__(None, None, None)
            self._reader = None
        
        self._is_open = False
    
    def __enter__(self) -> "LDMARKRuntime":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    # === Metadata Inspection ===
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def manifest(self) -> Manifest:
        """Get the artifact manifest."""
        if not self._reader:
            raise ArtifactCorruptedError("Artifact not open")
        return self._reader.manifest
    
    def get_format_version(self) -> ArtifactFormatVersion:
        return self.manifest.format_version
    
    def get_model_info(self) -> Optional[ModelInfo]:
        return self.manifest.model
    
    def get_compression_info(self) -> Optional[CompressionInfo]:
        return self.manifest.compression
    
    def get_tensor_count(self) -> int:
        return len(self.manifest.tensors)
    
    def list_tensors(self) -> List[str]:
        """List all tensor names in the artifact."""
        return [t.name for t in self.manifest.tensors]
    
    def get_tensor_info(self, name: str) -> Optional[TensorIndexEntry]:
        """Get tensor index entry (metadata only, no data loading)."""
        return self.manifest.get_tensor_index().get(name)
    
    # === Tensor Loading ===
    
    def get_tensor(self, name: str, decompress: bool = False) -> LoadedTensor:
        """
        Load a tensor by name.
        
        Args:
            name: Tensor name
            decompress: Whether to immediately decompress
            
        Returns:
            LoadedTensor with compressed data (and decompressed if requested)
            
        Raises:
            TensorNotFoundError: If tensor not in artifact
            ArtifactCorruptedError: If loading fails
        """
        if not self._is_open:
            raise ArtifactCorruptedError("Runtime not open")
        
        # Check if already loaded
        if name in self._loaded_tensors:
            tensor = self._loaded_tensors[name]
            if decompress and not tensor.is_decompressed:
                tensor.decompress(self.target_dtype)
                decomp_bytes = tensor.dequantized.decompressed_size_bytes
                compressed_bytes = tensor.entry.byte_length + tensor.entry.scale_byte_length
                self._memory.record_tensor_decompress(compressed_bytes, decomp_bytes)
            return tensor
        
        # Load from artifact
        if not self._reader:
            raise ArtifactCorruptedError("Reader not initialized")
        
        try:
            tensor_data: TensorData = self._reader.read_tensor(name)
        except KeyError as e:
            available = self.list_tensors()
            raise TensorNotFoundError(name, available) from e
        except Exception as e:
            raise ArtifactCorruptedError(f"Failed to load tensor {name}: {e}") from e
        
        # Track memory
        compressed_bytes = tensor_data.index_entry.byte_length + tensor_data.index_entry.scale_byte_length
        self._memory.record_tensor_load(compressed_bytes)
        
        # Create loaded tensor
        loaded = LoadedTensor(
            name=name,
            compressed_data=tensor_data.data,
            scales=tensor_data.scales,
            entry=tensor_data.index_entry,
        )
        
        # Decompress if requested
        if decompress:
            loaded.decompress(self.target_dtype)
            decomp_bytes = loaded.dequantized.decompressed_size_bytes
            self._memory.record_tensor_decompress(compressed_bytes, decomp_bytes)
        
        self._loaded_tensors[name] = loaded
        return loaded
    
    def get_tensor_data(self, name: str, target_dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Get decompressed tensor data directly.
        
        This is a convenience method that loads and decompresses in one call.
        """
        dtype = target_dtype or self.target_dtype
        tensor = self.get_tensor(name, decompress=True)
        return tensor.get_data(dtype)
    
    def release_tensor(self, name: str) -> None:
        """Release a loaded tensor to free memory."""
        if name in self._loaded_tensors:
            tensor = self._loaded_tensors[name]
            if tensor.is_decompressed:
                compressed = tensor.entry.byte_length + tensor.entry.scale_byte_length
                decomp = tensor.dequantized.decompressed_size_bytes if tensor.dequantized else 0
                self._memory.record_tensor_release(compressed, decomp)
                tensor.release_decompressed()
            else:
                compressed = tensor.entry.byte_length + tensor.entry.scale_byte_length
                self._memory.record_tensor_release(compressed, 0)
            del self._loaded_tensors[name]
    
    def release_all(self) -> None:
        """Release all loaded tensors."""
        for name in list(self._loaded_tensors.keys()):
            self.release_tensor(name)
    
    # === Computation ===
    
    def matmul(
        self,
        weight_name: str,
        input_tensor: np.ndarray,
        input_dtype: Optional[np.dtype] = None,
        output_dtype: Optional[np.dtype] = None,
    ) -> MatmulResult:
        """
        Matrix multiplication with a weight tensor.
        
        Reconstructs the weight tensor, then performs matmul.
        This is a correctness baseline - NOT optimized.
        
        Args:
            weight_name: Name of weight tensor in artifact
            input_tensor: Input array (will be converted to target_dtype)
            input_dtype: Optional dtype for input
            output_dtype: Optional dtype for output
            
        Returns:
            MatmulResult with output and timing info
        """
        start_time = time.perf_counter()
        
        # Load and decompress weight
        weight_tensor = self.get_tensor(weight_name, decompress=True)
        weight_data = weight_tensor.get_data(self.target_dtype)
        
        # Prepare input
        if input_dtype:
            input_data = input_tensor.astype(input_dtype)
        else:
            input_data = input_tensor.astype(self.target_dtype)
        
        # Track temporary memory
        input_bytes = input_data.nbytes
        weight_bytes = weight_data.nbytes
        self._memory.record_temporary_allocation(input_bytes + weight_bytes)
        
        try:
            # Perform matmul: input @ weight.T for [batch, in_features] @ [out_features, in_features].T = [batch, out_features]
            comp_start = time.perf_counter()
            output = np.matmul(input_data, weight_data.T)
            comp_time = (time.perf_counter() - comp_start) * 1000
            
            # Convert output dtype if needed
            if output_dtype:
                output = output.astype(output_dtype)
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            return MatmulResult(
                output=output,
                weight_name=weight_name,
                input_shape=input_data.shape,
                output_shape=output.shape,
                computation_time_ms=comp_time,
                memory_snapshot=self._memory.get_current(),
            )
        
        finally:
            self._memory.record_temporary_release(input_bytes + weight_bytes)
    
    def matmul_with_cached_weight(
        self,
        weight_name: str,
        input_tensor: np.ndarray,
        input_dtype: Optional[np.dtype] = None,
        output_dtype: Optional[np.dtype] = None,
    ) -> MatmulResult:
        """
        Matrix multiplication assuming weight is already loaded and decompressed.
        
        Use this when doing multiple matmuls with the same weight.
        """
        start_time = time.perf_counter()
        
        weight_tensor = self._loaded_tensors.get(weight_name)
        if not weight_tensor or not weight_tensor.is_decompressed:
            weight_tensor = self.get_tensor(weight_name, decompress=True)
        
        weight_data = weight_tensor.get_data(self.target_dtype)
        
        if input_dtype:
            input_data = input_tensor.astype(input_dtype)
        else:
            input_data = input_tensor.astype(self.target_dtype)
        
        input_bytes = input_data.nbytes
        weight_bytes = weight_data.nbytes
        self._memory.record_temporary_allocation(input_bytes + weight_bytes)
        
        try:
            # Perform matmul: input @ weight.T for [batch, in_features] @ [out_features, in_features].T = [batch, out_features]
            comp_start = time.perf_counter()
            output = np.matmul(input_data, weight_data.T)
            comp_time = (time.perf_counter() - comp_start) * 1000
            
            if output_dtype:
                output = output.astype(output_dtype)
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            return MatmulResult(
                output=output,
                weight_name=weight_name,
                input_shape=input_data.shape,
                output_shape=output.shape,
                computation_time_ms=comp_time,
                memory_snapshot=self._memory.get_current(),
            )
        
        finally:
            self._memory.record_temporary_release(input_bytes + weight_bytes)
    
    # === Memory Accounting ===
    
    @property
    def memory(self) -> MemoryAccountant:
        """Get memory accountant."""
        return self._memory
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        return self._memory.get_current()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage as dictionary."""
        return self._memory.to_dict()
    
    def get_peak_memory_estimate(self) -> int:
        """Get estimated peak memory usage."""
        return self._memory.get_peak_estimate()

    def get_peak_rss_bytes(self) -> int:
        """Measured process peak RSS in bytes (OS readback, not a counter)."""
        from src.ldmark.runtime.memory import get_process_rss_bytes
        return get_process_rss_bytes()

    def estimated_full_decompressed_bytes(self, target_dtype: Optional[np.dtype] = None) -> int:
        """Estimate runtime memory if ALL tensors were decompressed (weights only).

        STORAGE (compressed on disk) vs RUNTIME (decompressed in RAM) are
        deliberately separate: this returns the runtime side.
        """
        import numpy as np
        dtype = np.dtype(target_dtype or self.target_dtype)
        total = 0
        for t in self.manifest.tensors:
            n = 1
            for d in t.original_shape:
                n *= d
            total += n * dtype.itemsize
        return int(total)

    def check_budget(self, max_runtime_memory_bytes: int, label: str = "plan") -> Dict:
        """Check estimated full-decompressed runtime against a budget.

        Raises MemoryAccountingError with a clear message when over budget.
        Returns a verdict dict when within budget.
        """
        from src.ldmark.runtime.memory import format_bytes
        from .exceptions import MemoryAccountingError
        est = self.estimated_full_decompressed_bytes()
        if est > max_runtime_memory_bytes:
            over = est - max_runtime_memory_bytes
            raise MemoryAccountingError(
                f"REJECTED: {label} estimated runtime {format_bytes(est)} "
                f"exceeds budget {format_bytes(max_runtime_memory_bytes)} "
                f"by {format_bytes(over)}. Storage (compressed) != runtime "
                f"(decompressed): reduce tensors, context, or batch."
            )
        return {
            "fits": True,
            "estimated_runtime_bytes": est,
            "budget_bytes": max_runtime_memory_bytes,
            "label": label,
        }
    
    # === Benchmarking ===
    
    def benchmark_tensor(
        self,
        tensor_name: str,
        input_shape: Tuple[int, ...],
        num_warmup: int = 2,
        num_runs: int = 5,
    ) -> BenchmarkResult:
        """
        Benchmark loading, reconstruction, and matmul for a tensor.
        
        Args:
            tensor_name: Name of tensor to benchmark
            input_shape: Shape of input for matmul
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            
        Returns:
            BenchmarkResult with timing and memory stats
        """
        # Create test input
        weight_info = self.get_tensor_info(tensor_name)
        if not weight_info:
            raise TensorNotFoundError(tensor_name)
        
        # Input shape should be compatible with weight
        # Assume weight is [out_features, in_features] and input is [batch, in_features]
        in_features = weight_info.original_shape[-1]
        if input_shape[-1] != in_features:
            raise ComputationError(
                "matmul",
                f"Input shape {input_shape} incompatible with weight {weight_info.original_shape}"
            )
        
        # Warmup
        for _ in range(num_warmup):
            test_input = np.random.randn(*input_shape).astype(self.target_dtype)
            self.matmul(tensor_name, test_input)
        
        # Benchmark runs
        load_times = []
        reconstruct_times = []
        comp_times = []
        
        for _ in range(num_runs):
            # Time load
            load_start = time.perf_counter()
            tensor = self.get_tensor(tensor_name, decompress=False)
            load_time = (time.perf_counter() - load_start) * 1000
            load_times.append(load_time)
            
            # Time reconstruction
            recon_start = time.perf_counter()
            tensor.decompress(self.target_dtype)
            recon_time = (time.perf_counter() - recon_start) * 1000
            reconstruct_times.append(recon_time)
            
            # Time computation
            test_input = np.random.randn(*input_shape).astype(self.target_dtype)
            result = self.matmul_with_cached_weight(tensor_name, test_input)
            comp_times.append(result.computation_time_ms)
        
        avg_load = np.mean(load_times)
        avg_recon = np.mean(reconstruct_times)
        avg_comp = np.mean(comp_times)
        total = avg_load + avg_recon + avg_comp
        
        # Calculate ops/second (2 * M * N * K for matmul)
        weight_info = self.get_tensor_info(tensor_name)
        out_features, in_features = weight_info.original_shape[-2], weight_info.original_shape[-1]
        batch = input_shape[0]
        ops = 2 * batch * out_features * in_features
        ops_per_sec = ops / (avg_comp / 1000) if avg_comp > 0 else 0
        
        compressed_bytes = weight_info.byte_length + weight_info.scale_byte_length
        decompressed_bytes = np.prod(weight_info.original_shape) * np.dtype(self.target_dtype).itemsize
        
        return BenchmarkResult(
            tensor_name=tensor_name,
            load_time_ms=avg_load,
            reconstruction_time_ms=avg_recon,
            computation_time_ms=avg_comp,
            total_time_ms=total,
            compressed_bytes=compressed_bytes,
            decompressed_bytes=decompressed_bytes,
            peak_temporary_bytes=self._memory.peak_temporary_bytes,
            input_shape=input_shape,
            output_shape=(batch, out_features),
            ops_per_second=ops_per_sec,
        )
    
    # === Validation ===
    
    def validate_tensor(
        self,
        tensor_name: str,
        original_tensor: np.ndarray,
    ) -> CompressionMetrics:
        """
        Validate runtime reconstruction against original tensor.
        
        Uses existing compression error metrics.
        """
        reconstructed = self.get_tensor_data(tensor_name, target_dtype=np.float32)
        original = original_tensor.astype(np.float32)
        
        if original.shape != reconstructed.shape:
            raise ComputationError(
                "validate",
                f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
            )
        
        return calculate_error_metrics(original, reconstructed)
    
    # === Inspection ===
    
    def inspect(self) -> Dict:
        """Get full artifact inspection summary."""
        return {
            "format_version": self.manifest.format_version.value,
            "model": self.manifest.model.to_dict() if self.manifest.model else None,
            "compression": self.manifest.compression.to_dict() if self.manifest.compression else None,
            "tensor_count": self.get_tensor_count(),
            "tensor_names": self.list_tensors(),
            "loaded_tensors": list(self._loaded_tensors.keys()),
            "memory": self.get_memory_usage(),
            "target_dtype": str(self.target_dtype),
        }
    
    def print_memory_summary(self) -> None:
        """Print human-readable memory summary."""
        snap = self.get_memory_snapshot()
        from src.ldmark.runtime.memory import format_bytes
        print(f"LDMARK Runtime Memory Summary")
        print(f"  Compressed (storage):     {format_bytes(snap.compressed_weight_bytes)}")
        print(f"  Decompressed (runtime):   {format_bytes(snap.decompressed_weight_bytes)}")
        print(f"  Temporary (computation):  {format_bytes(snap.temporary_bytes)}")
        print(f"  Peak temporary:           {format_bytes(snap.peak_temporary_bytes)}")
        print(f"  Estimated peak total:     {format_bytes(snap.estimated_peak_bytes)}")
    
    # === Context Manager Support ===
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


def open_runtime(
    artifact_dir: Union[str, Path],
    verify_checksums: bool = True,
    mmap_tensors: bool = False,
    target_dtype: np.dtype = np.float16,
) -> LDMARKRuntime:
    """
    Convenience function to open an LDMARK runtime.
    
    Args:
        artifact_dir: Path to artifact directory
        verify_checksums: Whether to verify SHA-256 checksums
        mmap_tensors: Whether to memory-map tensor files
        target_dtype: Default dtype for dequantized tensors
        
    Returns:
        LDMARKRuntime instance
    """
    return LDMARKRuntime(artifact_dir, verify_checksums, mmap_tensors, target_dtype)