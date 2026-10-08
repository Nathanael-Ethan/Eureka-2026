"""
LDMARK Runtime - Memory Accounting

Tracks runtime memory usage distinguishing:
- Compressed artifact bytes (storage)
- Decompressed tensor bytes (runtime)
- Temporary computation memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import time


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    compressed_weight_bytes: int = 0
    decompressed_weight_bytes: int = 0
    temporary_bytes: int = 0
    peak_temporary_bytes: int = 0
    
    @property
    def estimated_peak_bytes(self) -> int:
        return max(
            self.compressed_weight_bytes + self.decompressed_weight_bytes + self.temporary_bytes,
            self.compressed_weight_bytes + self.decompressed_weight_bytes + self.peak_temporary_bytes
        )
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "compressed_weight_bytes": self.compressed_weight_bytes,
            "decompressed_weight_bytes": self.decompressed_weight_bytes,
            "temporary_bytes": self.temporary_bytes,
            "peak_temporary_bytes": self.peak_temporary_bytes,
            "estimated_peak_bytes": self.estimated_peak_bytes,
        }


@dataclass
class MemoryAccountant:
    """
    Tracks memory usage for runtime operations.
    
    Distinguishes between:
    - Compressed artifact bytes (what's on disk)
    - Decompressed tensor bytes (what's in memory after reconstruction)
    - Temporary computation memory (during matmul, etc.)
    """
    
    compressed_weight_bytes: int = 0
    decompressed_weight_bytes: int = 0
    temporary_bytes: int = 0
    peak_temporary_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _history: List[MemorySnapshot] = field(default_factory=list, repr=False)
    
    def record_tensor_load(self, compressed_bytes: int) -> None:
        """Record loading a compressed tensor from artifact."""
        with self._lock:
            self.compressed_weight_bytes += compressed_bytes
            self._snapshot()
    
    def record_tensor_decompress(self, compressed_bytes: int, decompressed_bytes: int) -> None:
        """Record tensor decompression."""
        with self._lock:
            # We already counted compressed bytes at load time
            self.decompressed_weight_bytes += decompressed_bytes
            self._snapshot()
    
    def record_tensor_release(self, compressed_bytes: int, decompressed_bytes: int) -> None:
        """Record tensor release (when no longer needed)."""
        with self._lock:
            self.compressed_weight_bytes = max(0, self.compressed_weight_bytes - compressed_bytes)
            self.decompressed_weight_bytes = max(0, self.decompressed_weight_bytes - decompressed_bytes)
            self._snapshot()
    
    def record_temporary_allocation(self, bytes_allocated: int) -> None:
        """Record temporary memory allocation (e.g., for matmul)."""
        with self._lock:
            self.temporary_bytes += bytes_allocated
            self.peak_temporary_bytes = max(self.peak_temporary_bytes, self.temporary_bytes)
            self._snapshot()
    
    def record_temporary_release(self, bytes_released: int) -> None:
        """Record temporary memory release."""
        with self._lock:
            self.temporary_bytes = max(0, self.temporary_bytes - bytes_released)
            self._snapshot()
    
    def _snapshot(self) -> None:
        """Record current memory state."""
        self._history.append(MemorySnapshot(
            timestamp=time.time(),
            compressed_weight_bytes=self.compressed_weight_bytes,
            decompressed_weight_bytes=self.decompressed_weight_bytes,
            temporary_bytes=self.temporary_bytes,
            peak_temporary_bytes=self.peak_temporary_bytes,
        ))
    
    def get_current(self) -> MemorySnapshot:
        """Get current memory snapshot."""
        return MemorySnapshot(
            timestamp=time.time(),
            compressed_weight_bytes=self.compressed_weight_bytes,
            decompressed_weight_bytes=self.decompressed_weight_bytes,
            temporary_bytes=self.temporary_bytes,
            peak_temporary_bytes=self.peak_temporary_bytes,
        )
    
    def get_peak_estimate(self) -> int:
        """Get estimated peak memory usage."""
        current = self.get_current()
        return current.estimated_peak_bytes
    
    def get_history(self) -> List[MemorySnapshot]:
        """Get memory history."""
        return list(self._history)
    
    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self.compressed_weight_bytes = 0
            self.decompressed_weight_bytes = 0
            self.temporary_bytes = 0
            self.peak_temporary_bytes = 0
            self._history.clear()
    
    def to_dict(self) -> Dict[str, int]:
        """Export current state as dictionary."""
        return self.get_current().to_dict()


def estimate_decompressed_size(
    tensor_shape: tuple,
    target_dtype: str = "float16"
) -> int:
    """
    Estimate decompressed tensor size in bytes.
    
    Args:
        tensor_shape: Original tensor shape
        target_dtype: Target dtype after decompression (float16, float32, etc.)
    
    Returns:
        Estimated size in bytes
    """
    import numpy as np
    dtype = np.dtype(target_dtype)
    num_elements = 1
    for dim in tensor_shape:
        num_elements *= dim
    return num_elements * dtype.itemsize


def estimate_memory_impact(
    compressed_size_bytes: int,
    tensor_shape: tuple,
    target_dtype: str = "float16"
) -> Dict[str, int]:
    """
    Estimate memory impact of loading and decompressing a tensor.
    
    Shows the difference between compressed storage and runtime memory.
    
    Args:
        compressed_size_bytes: Size of compressed tensor in artifact
        tensor_shape: Original tensor shape
        target_dtype: Target dtype after decompression
    
    Returns:
        Dictionary with memory estimates
    """
    decompressed = estimate_decompressed_size(tensor_shape, target_dtype)
    
    return {
        "compressed_bytes": compressed_size_bytes,
        "decompressed_bytes": decompressed,
        "expansion_factor": decompressed / compressed_size_bytes if compressed_size_bytes > 0 else 0,
        "total_if_fully_loaded": compressed_size_bytes + decompressed,
        "note": "This is the memory required if ALL tensors are fully decompressed. "
                "Lazy loading avoids this by keeping most tensors compressed.",
    }


def get_process_rss_bytes() -> int:
    """Return current process RSS in bytes (measured, not estimated).

    Uses resource.getrusage on POSIX, falls back to 0 when unavailable.
    This is MEASURED runtime memory, distinct from estimated counters.
    """
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes
        import platform
        if platform.system() == "Darwin":
            return int(rss)
        return int(rss * 1024)
    except Exception:
        return 0


def get_current_rss_bytes() -> int:
    """Best-effort current RSS (falls back to ru_maxrss peak on POSIX)."""
    return get_process_rss_bytes()


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def print_memory_comparison(
    compressed_size_bytes: int,
    tensor_shape: tuple,
    target_dtype: str = "float16"
) -> None:
    """
    Print a clear comparison showing storage vs runtime memory.
    
    This makes the distinction between STORAGE COMPRESSION
    and RUNTIME DECOMPRESSION obvious.
    """
    estimates = estimate_memory_impact(compressed_size_bytes, tensor_shape, target_dtype)
    
    print("=" * 60)
    print("LDMARK MEMORY ACCOUNTING")
    print("=" * 60)
    print(f"Compressed (storage):     {format_bytes(estimates['compressed_bytes']):>12}")
    print(f"Decompressed (runtime):   {format_bytes(estimates['decompressed_bytes']):>12}")
    print(f"Expansion factor:         {estimates['expansion_factor']:>10.1f}x")
    print(f"Total if fully loaded:    {format_bytes(estimates['total_if_fully_loaded']):>12}")
    print("-" * 60)
    print("NOTE: This is STORAGE COMPRESSION, not direct low-bit computation.")
    print("      The tensor is fully expanded to FP16/FP32 for computation.")
    print("      Future: direct computation on compressed representations.")
    print("=" * 60)