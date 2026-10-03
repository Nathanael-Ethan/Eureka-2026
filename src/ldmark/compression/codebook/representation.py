"""
Codebook representation data structures for LDMARK experimental compression.

Defines the core data structures for codebook-based quantization:
- CodebookConfig: Configuration for codebook quantization
- CodebookTensor: Container for codebook-quantized tensor data
- CodebookGroup: Group-wise codebook structure
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import uuid


@dataclass(frozen=True)
class CodebookConfig:
    """Configuration for codebook quantization."""
    codebook_size: int
    group_size: Optional[int] = None  # None = global codebook, int = group-wise
    max_iterations: int = 20
    tolerance: float = 1e-4
    seed: int = 42
    initialization: str = "percentile"  # "uniform", "percentile", "random", "kmeans++"
    
    def __post_init__(self):
        if self.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2")
        if self.codebook_size > 256:
            raise ValueError("codebook_size must be <= 256 (8-bit index limit)")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError("group_size must be positive if specified")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.initialization not in ("uniform", "percentile", "random", "kmeans++"):
            raise ValueError(f"Unknown initialization: {self.initialization}")
    
    @property
    def index_bit_width(self) -> int:
        """Minimum bits needed to represent indices."""
        return max(1, int(np.ceil(np.log2(self.codebook_size))))
    
    @property
    def is_group_wise(self) -> bool:
        """Whether this config uses group-wise codebooks."""
        return self.group_size is not None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "codebook_size": self.codebook_size,
            "group_size": self.group_size,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "seed": self.seed,
            "initialization": self.initialization,
            "index_bit_width": self.index_bit_width,
            "is_group_wise": self.is_group_wise,
        }


@dataclass(frozen=True)
class CodebookGroup:
    """A single codebook group with its indices."""
    codebook: np.ndarray  # Shape: (codebook_size,)
    indices: np.ndarray   # Shape: (group_size,) or (remaining_size,)
    start_idx: int        # Start index in original flattened tensor
    end_idx: int          # End index (exclusive)
    
    def __post_init__(self):
        if self.codebook.ndim != 1:
            raise ValueError("codebook must be 1D")
        if self.indices.ndim != 1:
            raise ValueError("indices must be 1D")
        if len(self.indices) != (self.end_idx - self.start_idx):
            raise ValueError("indices length must match group size")
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.codebook)):
            raise ValueError("indices out of bounds for codebook")
    
    @property
    def size(self) -> int:
        return len(self.indices)
    
    @property
    def codebook_size(self) -> int:
        return len(self.codebook)
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the group values from codebook and indices."""
        return self.codebook[self.indices]


@dataclass(frozen=True)
class CodebookTensor:
    """Container for codebook-quantized tensor data."""
    groups: List[CodebookGroup]
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    config: CodebookConfig
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def __post_init__(self):
        if not self.groups:
            raise ValueError("At least one group required")
        expected_size = np.prod(self.original_shape)
        actual_size = sum(g.size for g in self.groups)
        if actual_size != expected_size:
            raise ValueError(f"Group sizes sum to {actual_size}, expected {expected_size}")
        # Verify all groups have same codebook size
        codebook_sizes = {g.codebook_size for g in self.groups}
        if len(codebook_sizes) != 1:
            raise ValueError("All groups must have same codebook size")
    
    @property
    def total_elements(self) -> int:
        return np.prod(self.original_shape)
    
    @property
    def num_groups(self) -> int:
        return len(self.groups)
    
    @property
    def codebook_size(self) -> int:
        return self.groups[0].codebook_size
    
    @property
    def index_bit_width(self) -> int:
        return self.config.index_bit_width
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct full tensor from all groups."""
        flat = np.zeros(self.total_elements, dtype=np.float32)
        for group in self.groups:
            flat[group.start_idx:group.end_idx] = group.reconstruct()
        return flat.reshape(self.original_shape)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "original_shape": self.original_shape,
            "original_dtype": str(self.original_dtype),
            "config": self.config.to_dict(),
            "num_groups": self.num_groups,
            "codebook_size": self.codebook_size,
            "index_bit_width": self.index_bit_width,
            "total_elements": self.total_elements,
        }


def create_codebook_tensor(
    tensor: np.ndarray,
    config: CodebookConfig,
    codebooks: Optional[List[np.ndarray]] = None,
    indices: Optional[List[np.ndarray]] = None,
) -> CodebookTensor:
    """
    Create a CodebookTensor from tensor and codebook data.
    
    Args:
        tensor: Original tensor (used for shape/dtype only)
        config: Codebook configuration
        codebooks: List of codebook arrays (one per group)
        indices: List of index arrays (one per group)
        
    Returns:
        CodebookTensor with groups populated
    """
    flat = tensor.flatten()
    total_elements = len(flat)
    
    if config.is_group_wise:
        group_size = config.group_size
        num_groups = (total_elements + group_size - 1) // group_size
    else:
        group_size = total_elements
        num_groups = 1
    
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, total_elements)
        group_len = end - start
        
        if codebooks is not None and indices is not None:
            cb = codebooks[g] if config.is_group_wise else codebooks[0]
            idx = indices[g] if config.is_group_wise else indices[0][start:end]
        else:
            # Will be populated later by quantization
            cb = np.zeros(config.codebook_size, dtype=np.float32)
            idx = np.zeros(group_len, dtype=np.uint8)
        
        groups.append(CodebookGroup(
            codebook=cb,
            indices=idx,
            start_idx=start,
            end_idx=end,
        ))
    
    return CodebookTensor(
        groups=groups,
        original_shape=tensor.shape,
        original_dtype=tensor.dtype,
        config=config,
    )


def reconstruct_from_codebook(cb_tensor: CodebookTensor) -> np.ndarray:
    """Reconstruct tensor from CodebookTensor."""
    return cb_tensor.reconstruct()