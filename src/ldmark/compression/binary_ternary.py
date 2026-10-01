"""
Binary and ternary representation experimental abstractions for LDMARK.

These are research scaffolding - NOT production implementations.
Purpose: enable future experimentation with {-1, +1} and {-1, 0, +1} representations.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np


class BinaryEncoding(Enum):
    """Encoding schemes for binary weights."""
    SIGN = "sign"           # {-1, +1} -> {0, 1}
    ZERO_ONE = "zero_one"   # {0, 1}


class TernaryEncoding(Enum):
    """Encoding schemes for ternary weights."""
    MINUS_ZERO_PLUS = "minus_zero_plus"  # {-1, 0, +1} -> {0, 1, 2}
    BALANCED = "balanced"                 # {-1, 0, +1} with 2-bit encoding


@dataclass(frozen=True)
class BinaryConfig:
    """Configuration for binary weight representation."""
    encoding: BinaryEncoding = BinaryEncoding.SIGN
    scale_dtype: np.dtype = np.float16
    group_size: int = 128
    
    def bits_per_weight(self) -> float:
        """Theoretical bits per weight for binary (1 bit + scale overhead)."""
        return 1.0
    
    def scale_overhead_bits_per_weight(self) -> float:
        """Scale overhead in bits per weight."""
        return (np.dtype(self.scale_dtype).itemsize * 8) / self.group_size


@dataclass(frozen=True)
class TernaryConfig:
    """Configuration for ternary weight representation."""
    encoding: TernaryEncoding = TernaryEncoding.MINUS_ZERO_PLUS
    scale_dtype: np.dtype = np.float16
    group_size: int = 128
    
    def bits_per_weight(self) -> float:
        """Theoretical bits per weight for ternary (~1.585 bits = log2(3))."""
        return np.log2(3)  # ~1.585
    
    def scale_overhead_bits_per_weight(self) -> float:
        """Scale overhead in bits per weight."""
        return (np.dtype(self.scale_dtype).itemsize * 8) / self.group_size


@dataclass(frozen=True)
class BinaryTensor:
    """Container for binary quantized tensor."""
    packed_data: np.ndarray  # uint8 array
    scales: np.ndarray       # scale per group
    config: BinaryConfig
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    num_weights: int
    
    def __post_init__(self):
        expected_packed_bytes = (self.num_weights + 7) // 8
        if len(self.packed_data) != expected_packed_bytes:
            raise ValueError(f"Packed data size mismatch")


@dataclass(frozen=True)
class TernaryTensor:
    """Container for ternary quantized tensor."""
    packed_data: np.ndarray  # uint8 array (2 bits per weight)
    scales: np.ndarray       # scale per group
    config: TernaryConfig
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    num_weights: int
    
    def __post_init__(self):
        expected_packed_bytes = (self.num_weights * 2 + 7) // 8
        if len(self.packed_data) != expected_packed_bytes:
            raise ValueError(f"Packed data size mismatch")


def _to_dtype(scale_dtype) -> np.dtype:
    """Convert to numpy dtype."""
    return np.dtype(scale_dtype)


def calculate_binary_storage(
    num_weights: int,
    config: BinaryConfig
) -> int:
    """
    Calculate storage size in bytes for binary representation.
    
    Formula: num_weights bits + num_groups * scale_bits
    """
    weight_bits = num_weights * 1  # 1 bit per weight
    num_groups = (num_weights + config.group_size - 1) // config.group_size
    scale_bits = num_groups * _to_dtype(config.scale_dtype).itemsize * 8
    total_bits = weight_bits + scale_bits
    return int(np.ceil(total_bits / 8))


def calculate_ternary_storage(
    num_weights: int,
    config: TernaryConfig
) -> int:
    """
    Calculate storage size in bytes for ternary representation.
    
    Formula: num_weights * log2(3) bits + num_groups * scale_bits
    """
    weight_bits = num_weights * np.log2(3)  # ~1.585 bits per weight
    num_groups = (num_weights + config.group_size - 1) // config.group_size
    scale_bits = num_groups * _to_dtype(config.scale_dtype).itemsize * 8
    total_bits = weight_bits + scale_bits
    return int(np.ceil(total_bits / 8))


def binary_quantize_sign(
    tensor: np.ndarray,
    config: BinaryConfig
) -> BinaryTensor:
    """
    Quantize tensor to binary {-1, +1} using sign function.
    
    This is a RESEARCH FUNCTION - not optimized for quality.
    """
    flat = tensor.flatten()
    n_weights = len(flat)
    n_groups = (n_weights + config.group_size - 1) // config.group_size
    
    # Compute scales per group
    scale_dtype = _to_dtype(config.scale_dtype)
    scales = np.zeros(n_groups, dtype=scale_dtype)
    signs = np.zeros(n_weights, dtype=np.uint8)
    
    for g in range(n_groups):
        start = g * config.group_size
        end = min(start + config.group_size, n_weights)
        group = flat[start:end]
        
        # Scale as mean absolute value
        scale = np.mean(np.abs(group))
        if scale == 0:
            scale = 1.0
        scales[g] = scale_dtype.type(scale)
        
        # Binary sign: -1 -> 0, +1 -> 1
        signs[start:end] = (group >= 0).astype(np.uint8)
    
    # Pack bits
    packed_bytes = (n_weights + 7) // 8
    packed = np.zeros(packed_bytes, dtype=np.uint8)
    for i in range(n_weights):
        if signs[i]:
            packed[i // 8] |= (1 << (i % 8))
    
    return BinaryTensor(
        packed_data=packed,
        scales=scales,
        config=config,
        original_shape=tensor.shape,
        original_dtype=tensor.dtype,
        num_weights=n_weights,
    )


def binary_dequantize(btensor: BinaryTensor) -> np.ndarray:
    """Dequantize binary tensor (research only)."""
    config = btensor.config
    n_weights = btensor.num_weights
    n_groups = len(btensor.scales)
    
    dequantized = np.zeros(n_weights, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * config.group_size
        end = min(start + config.group_size, n_weights)
        scale = float(btensor.scales[g])
        
        for i in range(start, end):
            byte_idx = i // 8
            bit_idx = i % 8
            bit = (btensor.packed_data[byte_idx] >> bit_idx) & 1
            # 0 -> -1, 1 -> +1
            dequantized[i] = scale * (1.0 if bit else -1.0)
    
    return dequantized.reshape(btensor.original_shape)


def ternary_quantize_threshold(
    tensor: np.ndarray,
    config: TernaryConfig,
    threshold: float = 0.05
) -> TernaryTensor:
    """
    Quantize tensor to ternary {-1, 0, +1} using threshold.
    
    This is a RESEARCH FUNCTION - not optimized for quality.
    """
    flat = tensor.flatten()
    n_weights = len(flat)
    n_groups = (n_weights + config.group_size - 1) // config.group_size
    
    scale_dtype = _to_dtype(config.scale_dtype)
    scales = np.zeros(n_groups, dtype=scale_dtype)
    # 2 bits per weight: 00=-1, 01=0, 10=+1 (11 unused)
    packed_bytes = (n_weights * 2 + 7) // 8
    packed = np.zeros(packed_bytes, dtype=np.uint8)
    
    for g in range(n_groups):
        start = g * config.group_size
        end = min(start + config.group_size, n_weights)
        group = flat[start:end]
        
        # Scale as mean absolute value of non-zero elements
        abs_group = np.abs(group)
        scale = np.mean(abs_group[abs_group > threshold])
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        scales[g] = scale_dtype.type(scale)
        
        # Ternary quantization
        for i, val in enumerate(group):
            idx = start + i
            if val > threshold:
                trit = 2  # +1 -> 10
            elif val < -threshold:
                trit = 0  # -1 -> 00
            else:
                trit = 1  # 0 -> 01
            
            bit_pos = idx * 2
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            
            if bit_offset <= 6:
                packed[byte_idx] |= (trit << bit_offset)
            else:
                # Spans byte boundary
                packed[byte_idx] |= (trit << bit_offset) & 0xFF
                packed[byte_idx + 1] |= (trit >> (8 - bit_offset)) & 0xFF
    
    return TernaryTensor(
        packed_data=packed,
        scales=scales,
        config=config,
        original_shape=tensor.shape,
        original_dtype=tensor.dtype,
        num_weights=n_weights,
    )


def ternary_dequantize(ttensor: TernaryTensor) -> np.ndarray:
    """Dequantize ternary tensor (research only)."""
    config = ttensor.config
    n_weights = ttensor.num_weights
    n_groups = len(ttensor.scales)
    
    dequantized = np.zeros(n_weights, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * config.group_size
        end = min(start + config.group_size, n_weights)
        scale = float(ttensor.scales[g])
        
        for i in range(start, end):
            bit_pos = i * 2
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            
            if bit_offset <= 6:
                trit = (ttensor.packed_data[byte_idx] >> bit_offset) & 0x3
            else:
                trit = ((ttensor.packed_data[byte_idx] >> bit_offset) | 
                       (ttensor.packed_data[byte_idx + 1] << (8 - bit_offset))) & 0x3
            
            # 00=-1, 01=0, 10=+1
            if trit == 0:
                dequantized[i] = -scale
            elif trit == 1:
                dequantized[i] = 0.0
            else:  # trit == 2
                dequantized[i] = scale
    
    return dequantized.reshape(ttensor.original_shape)