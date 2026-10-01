"""
PrismML Q1_0_g128 storage calculator for LDMARK.

This module reproduces the theoretical storage mathematics behind PrismML's
Q1_0_g128 representation. It is a RESEARCH CALCULATOR ONLY - not an implementation
of PrismML.

PrismML Q1_0_g128 representation:
- 128 one-bit weights (128 bits)
- 1 FP16 scale (16 bits)
- Total: 144 bits for 128 weights = 1.125 bits/weight
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class PrismMLConfig:
    """Configuration for PrismML-style storage calculation."""
    group_size: int = 128
    weight_bits: int = 1        # 1-bit weights
    scale_bits: int = 16        # FP16 scale
    model_params: int = 27_000_000_000  # 27B parameters
    
    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.weight_bits <= 0:
            raise ValueError("weight_bits must be positive")
        if self.scale_bits <= 0:
            raise ValueError("scale_bits must be positive")
        if self.model_params <= 0:
            raise ValueError("model_params must be positive")


def prismml_q1_0_g128_bits_per_weight(
    group_size: int = 128,
    weight_bits: int = 1,
    scale_bits: int = 16
) -> float:
    """
    Calculate theoretical bits per weight for PrismML Q1_0_g128.
    
    Formula: (group_size * weight_bits + scale_bits) / group_size
    
    For g128: (128 * 1 + 16) / 128 = 144 / 128 = 1.125 bits/weight
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    total_bits = group_size * weight_bits + scale_bits
    return total_bits / group_size


def calculate_prismml_storage(
    config: PrismMLConfig
) -> dict:
    """
    Calculate storage requirements for PrismML-style representation.
    
    Returns dictionary with:
    - bits_per_weight
    - total_bits
    - total_bytes
    - total_gb
    - group_count
    """
    bpw = prismml_q1_0_g128_bits_per_weight(
        config.group_size, config.weight_bits, config.scale_bits
    )
    
    total_bits = config.model_params * bpw
    total_bytes = total_bits / 8
    total_gb = total_bytes / (1024 ** 3)
    group_count = (config.model_params + config.group_size - 1) // config.group_size
    
    return {
        "bits_per_weight": bpw,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "group_count": group_count,
        "weight_bits": config.weight_bits,
        "scale_bits": config.scale_bits,
        "group_size": config.group_size,
        "model_params": config.model_params,
    }


def calculate_generic_groupwise_storage(
    model_params: int,
    weight_bits: int,
    group_size: int,
    scale_bits: int = 16
) -> dict:
    """
    Generic group-wise storage calculator.
    
    Can be used for any group-wise quantization scheme.
    """
    bpw = (group_size * weight_bits + scale_bits) / group_size
    total_bits = model_params * bpw
    total_bytes = total_bits / 8
    total_gb = total_bytes / (1024 ** 3)
    group_count = (model_params + group_size - 1) // group_size
    
    return {
        "bits_per_weight": bpw,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "total_gb": total_gb,
        "group_count": group_count,
        "weight_bits": weight_bits,
        "scale_bits": scale_bits,
        "group_size": group_size,
        "model_params": model_params,
    }


def compare_representations(
    model_params: int = 27_000_000_000
) -> dict:
    """
    Compare various quantization representations for a given model size.
    
    Returns comparison table of storage requirements.
    """
    representations = [
        ("FP32", 32, 0, 0),           # No grouping, no scale
        ("FP16", 16, 0, 0),           # No grouping, no scale
        ("INT8", 8, 128, 16),         # Group-wise INT8
        ("INT4", 4, 128, 16),         # Group-wise INT4
        ("PrismML Q1_0_g128", 1, 128, 16),
        ("Binary (1-bit)", 1, 128, 16),
        ("Ternary (~1.585-bit)", np.log2(3), 128, 16),
    ]
    
    results = []
    for name, w_bits, g_size, s_bits in representations:
        if g_size == 0:
            bpw = w_bits
            total_bits = model_params * bpw
            group_count = 0
        else:
            bpw = (g_size * w_bits + s_bits) / g_size
            total_bits = model_params * bpw
            group_count = (model_params + g_size - 1) // g_size
        
        total_gb = total_bits / 8 / (1024 ** 3)
        results.append({
            "representation": name,
            "bits_per_weight": bpw,
            "total_gb": total_gb,
            "group_count": group_count,
        })
    
    return {
        "model_params": model_params,
        "comparisons": results,
    }


def validate_prismml_claim(
    claimed_gb: float = 3.9,
    model_params: int = 27_000_000_000,
    tolerance: float = 0.1
) -> dict:
    """
    Validate PrismML's claimed 3.9 GB for 27B model at Q1_0_g128.
    
    Returns validation result with calculated vs claimed values.
    """
    calc = calculate_prismml_storage(PrismMLConfig(model_params=model_params))
    calculated_gb = calc["total_gb"]
    difference = abs(calculated_gb - claimed_gb)
    relative_error = difference / claimed_gb
    
    return {
        "claimed_gb": claimed_gb,
        "calculated_gb": calculated_gb,
        "difference_gb": difference,
        "relative_error": relative_error,
        "within_tolerance": relative_error <= tolerance,
        "tolerance": tolerance,
        "details": calc,
    }