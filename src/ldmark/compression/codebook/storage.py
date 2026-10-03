"""
Storage accounting for codebook-based quantization.

Calculates actual storage requirements including indices, codebooks, and metadata.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .representation import CodebookTensor, CodebookConfig, CodebookGroup


@dataclass(frozen=True)
class StorageAccounting:
    """Detailed storage breakdown for codebook representation."""
    index_bytes: int
    codebook_bytes: int
    metadata_bytes: int
    total_bytes: int
    theoretical_bits_per_weight: float
    effective_bits_per_weight: float
    original_bytes: int
    compression_ratio: float
    group_count: int
    codebook_size: int
    index_bit_width: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_bytes": self.index_bytes,
            "codebook_bytes": self.codebook_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "theoretical_bits_per_weight": self.theoretical_bits_per_weight,
            "effective_bits_per_weight": self.effective_bits_per_weight,
            "original_bytes": self.original_bytes,
            "compression_ratio": self.compression_ratio,
            "group_count": self.group_count,
            "codebook_size": self.codebook_size,
            "index_bit_width": self.index_bit_width,
        }


def calculate_codebook_storage(
    cb_tensor: CodebookTensor,
    metadata_bytes: int = 0,
) -> StorageAccounting:
    """
    Calculate actual storage for a codebook-quantized tensor.
    
    Accounts for:
    - Index storage (packed to minimum bit width)
    - Codebook storage (float32 per entry)
    - Group metadata (if group-wise)
    - General metadata
    
    Args:
        cb_tensor: CodebookTensor to analyze
        metadata_bytes: Additional metadata overhead
        
    Returns:
        StorageAccounting with complete breakdown
    """
    config = cb_tensor.config
    total_elements = cb_tensor.total_elements
    num_groups = cb_tensor.num_groups
    codebook_size = cb_tensor.codebook_size
    index_bit_width = cb_tensor.index_bit_width
    
    # Index storage: pack indices to minimum bit width
    # For 1-bit: 1 bit per index
    # For 2-bit: 2 bits per index (pack 4 per byte)
    # For 4-bit: 4 bits per index (pack 2 per byte)
    # For 8-bit: 1 byte per index
    
    if index_bit_width == 1:
        index_bytes = (total_elements + 7) // 8
    elif index_bit_width == 2:
        index_bytes = (total_elements * 2 + 7) // 8
    elif index_bit_width == 4:
        index_bytes = (total_elements * 4 + 7) // 8
    elif index_bit_width == 8:
        index_bytes = total_elements
    else:
        # Round up to next byte boundary
        index_bytes = (total_elements * index_bit_width + 7) // 8
    
    # Codebook storage
    if config.is_group_wise:
        # One codebook per group
        codebook_bytes = num_groups * codebook_size * 4  # float32
    else:
        # Single global codebook
        codebook_bytes = codebook_size * 4
    
    total_bytes = index_bytes + codebook_bytes + metadata_bytes
    
    # Theoretical bits/weight (index bits only, ignoring overhead)
    theoretical_bpw = index_bit_width
    
    # Effective bits/weight (including all overhead)
    effective_bpw = (total_bytes * 8) / total_elements if total_elements > 0 else 0
    
    # Original bytes (assuming float32)
    original_bytes = total_elements * 4
    
    compression_ratio = original_bytes / total_bytes if total_bytes > 0 else float('inf')
    
    return StorageAccounting(
        index_bytes=index_bytes,
        codebook_bytes=codebook_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=total_bytes,
        theoretical_bits_per_weight=theoretical_bpw,
        effective_bits_per_weight=effective_bpw,
        original_bytes=original_bytes,
        compression_ratio=compression_ratio,
        group_count=num_groups,
        codebook_size=codebook_size,
        index_bit_width=index_bit_width,
    )


def calculate_effective_bits_per_weight(
    total_elements: int,
    index_bit_width: int,
    num_codebooks: int,
    codebook_size: int,
    metadata_bytes: int = 0,
) -> float:
    """
    Calculate effective bits per weight including all overhead.
    
    Args:
        total_elements: Total number of weight elements
        index_bit_width: Bits per index
        num_codebooks: Number of codebooks (1 for global, groups for group-wise)
        codebook_size: Size of each codebook
        metadata_bytes: Additional metadata bytes
        
    Returns:
        Effective bits per weight
    """
    index_bits = total_elements * index_bit_width
    codebook_bits = num_codebooks * codebook_size * 32  # float32 = 32 bits
    metadata_bits = metadata_bytes * 8
    total_bits = index_bits + codebook_bits + metadata_bits
    return total_bits / total_elements if total_elements > 0 else 0


def calculate_uniform_storage_comparison(
    original_bytes: int,
    target_bits: int,
    group_size: Optional[int] = None,
    scale_bits: int = 16,
) -> Dict[str, Any]:
    """
    Calculate storage for uniform quantization for comparison.
    
    Args:
        original_bytes: Original tensor size in bytes
        target_bits: Target bits per weight (e.g., 4 for INT4)
        group_size: Group size for group-wise quantization (None for none)
        scale_bits: Bits per scale factor
        
    Returns:
        Dictionary with uniform storage breakdown
    """
    total_elements = original_bytes // 4  # Assuming float32
    
    if group_size is None:
        # No grouping
        num_groups = 0
        scale_bits_total = 0
    else:
        num_groups = (total_elements + group_size - 1) // group_size
        scale_bits_total = num_groups * scale_bits
    
    weight_bits = total_elements * target_bits
    total_bits = weight_bits + scale_bits_total
    total_bytes = (total_bits + 7) // 8
    
    return {
        "total_bytes": total_bytes,
        "weight_bytes": (weight_bits + 7) // 8,
        "scale_bytes": (scale_bits_total + 7) // 8,
        "bits_per_weight": total_bits / total_elements if total_elements > 0 else 0,
        "theoretical_bpw": target_bits + (scale_bits / group_size if group_size else 0),
        "compression_ratio": original_bytes / total_bytes if total_bytes > 0 else float('inf'),
    }


def calculate_codebook_storage_breakdown(
    tensor_shape: Tuple[int, ...],
    config: CodebookConfig,
    metadata_bytes: int = 0,
) -> Dict[str, Any]:
    """
    Calculate expected storage for a tensor with given config (without actual data).
    
    Args:
        tensor_shape: Shape of the tensor
        config: Codebook configuration
        metadata_bytes: Additional metadata
        
    Returns:
        Storage breakdown dictionary
    """
    total_elements = np.prod(tensor_shape)
    index_bit_width = config.index_bit_width
    
    if config.is_group_wise:
        group_size = config.group_size
        num_groups = (total_elements + group_size - 1) // group_size
    else:
        group_size = total_elements
        num_groups = 1
    
    # Index storage
    if index_bit_width == 1:
        index_bytes = (total_elements + 7) // 8
    elif index_bit_width == 2:
        index_bytes = (total_elements * 2 + 7) // 8
    elif index_bit_width == 4:
        index_bytes = (total_elements * 4 + 7) // 8
    elif index_bit_width == 8:
        index_bytes = total_elements
    else:
        index_bytes = (total_elements * index_bit_width + 7) // 8
    
    # Codebook storage
    codebook_bytes = num_groups * config.codebook_size * 4
    
    total_bytes = index_bytes + codebook_bytes + metadata_bytes
    original_bytes = total_elements * 4
    
    return {
        "original_bytes": original_bytes,
        "index_bytes": index_bytes,
        "codebook_bytes": codebook_bytes,
        "metadata_bytes": metadata_bytes,
        "total_bytes": total_bytes,
        "effective_bpw": (total_bytes * 8) / total_elements if total_elements > 0 else 0,
        "theoretical_bpw": index_bit_width,
        "compression_ratio": original_bytes / total_bytes if total_bytes > 0 else float('inf'),
        "num_groups": num_groups,
        "group_size": group_size,
        "codebook_size": config.codebook_size,
        "index_bit_width": index_bit_width,
    }