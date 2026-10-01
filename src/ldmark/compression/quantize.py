"""
Group-wise quantization implementation for LDMARK compression laboratory.

Supports FP32/FP16 -> INT8/INT4 quantization with group-wise scaling.
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Optional, Tuple


class QuantizationTarget(Enum):
    """Target quantization bit-width."""
    INT8 = 8
    INT4 = 4


class SourceDType(Enum):
    """Source tensor data type."""
    FP32 = np.float32
    FP16 = np.float16


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for group-wise quantization."""
    target_bits: QuantizationTarget = QuantizationTarget.INT8
    group_size: int = 128
    scale_dtype: np.dtype = np.float16
    symmetric: bool = True
    
    def __post_init__(self):
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.target_bits not in (QuantizationTarget.INT8, QuantizationTarget.INT4):
            raise ValueError("target_bits must be INT8 or INT4")


@dataclass(frozen=True)
class QuantizedTensor:
    """Container for quantized tensor data."""
    data: np.ndarray
    scales: np.ndarray
    config: QuantizationConfig
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    
    def __post_init__(self):
        if self.data.ndim != 1:
            raise ValueError("Quantized data must be 1D")
        if self.scales.ndim != 1:
            raise ValueError("Scales must be 1D")
        expected_groups = (np.prod(self.original_shape) + self.config.group_size - 1) // self.config.group_size
        if len(self.scales) != expected_groups:
            raise ValueError(f"Expected {expected_groups} scales, got {len(self.scales)}")


def calculate_scale(
    group: np.ndarray,
    target_bits: QuantizationTarget,
    symmetric: bool = True,
    scale_dtype: np.dtype = np.float16
) -> np.floating:
    """
    Calculate quantization scale for a group of values.
    
    For symmetric quantization: scale = max(abs(group)) / (2^(bits-1) - 1)
    For asymmetric quantization: scale = (max - min) / (2^bits - 1)
    """
    scale_dtype = np.dtype(scale_dtype)
    if len(group) == 0:
        return scale_dtype.type(1.0)
    
    max_val = np.max(group)
    min_val = np.min(group)
    
    if symmetric:
        max_abs = max(abs(max_val), abs(min_val))
        if max_abs == 0:
            return scale_dtype.type(1.0)
        qmax = (1 << (target_bits.value - 1)) - 1
        scale = max_abs / qmax
    else:
        qmax = (1 << target_bits.value) - 1
        if max_val == min_val:
            return scale_dtype.type(1.0)
        scale = (max_val - min_val) / qmax
    
    return scale_dtype.type(scale)


def _pack_int4(values: np.ndarray) -> np.ndarray:
    """Pack two INT4 values (each in [-8, 7]) into one uint8."""
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    for i in range(0, n, 2):
        v0 = np.clip(values[i], -8, 7) & 0xF
        v1 = np.clip(values[i + 1], -8, 7) & 0xF if i + 1 < n else 0
        packed[i // 2] = (v0 << 4) | v1
    return packed


def _unpack_int4(packed: np.ndarray, n_values: int) -> np.ndarray:
    """Unpack INT4 values from uint8 array."""
    if n_values == 0:
        return np.zeros(0, dtype=np.int8)
    values = np.zeros(n_values, dtype=np.int8)
    for i in range(n_values):
        byte_idx = i // 2
        if i % 2 == 0:
            v = int((packed[byte_idx] >> 4) & 0xF)
        else:
            v = int(packed[byte_idx] & 0xF)
        # Sign extend from 4 bits
        if v >= 8:
            v -= 16
        values[i] = v
    return values


def quantize_groupwise(
    tensor: np.ndarray,
    config: QuantizationConfig
) -> QuantizedTensor:
    """
    Quantize a tensor using group-wise quantization.
    
    Args:
        tensor: Input tensor (any shape)
        config: Quantization configuration
        
    Returns:
        QuantizedTensor containing quantized data and scales
    """
    original_shape = tensor.shape
    original_dtype = tensor.dtype
    flat = tensor.flatten().astype(np.float32)
    n_elements = len(flat)
    n_groups = (n_elements + config.group_size - 1) // config.group_size
    
    # Determine target integer type and packing
    if config.target_bits == QuantizationTarget.INT8:
        target_dtype = np.int8
        qmax = 127
        qmin = -128 if config.symmetric else 0
        use_packing = False
    else:  # INT4
        target_dtype = np.uint8  # Packed INT4 storage
        qmax = 7
        qmin = -8 if config.symmetric else 0
        use_packing = True
    
    quantized_data_list = []
    scales = np.zeros(n_groups, dtype=config.scale_dtype)
    
    for g in range(n_groups):
        start = g * config.group_size
        end = min(start + config.group_size, n_elements)
        group = flat[start:end]
        
        scale = calculate_scale(group, config.target_bits, config.symmetric, config.scale_dtype)
        scales[g] = scale
        
        if scale == 0:
            quantized = np.zeros(len(group), dtype=np.int8)
        else:
            # Quantize
            quantized = np.round(group / scale).astype(np.int32)
            quantized = np.clip(quantized, qmin, qmax)
        
        if use_packing:
            quantized_data_list.append(_pack_int4(quantized))
        else:
            quantized_data_list.append(quantized.astype(np.int8))
    
    # Concatenate all groups
    if use_packing:
        quantized_data = np.concatenate(quantized_data_list)
    else:
        quantized_data = np.concatenate(quantized_data_list)
    
    return QuantizedTensor(
        data=quantized_data,
        scales=scales,
        config=config,
        original_shape=original_shape,
        original_dtype=original_dtype
    )


def dequantize_groupwise(qtensor: QuantizedTensor) -> np.ndarray:
    """
    Dequantize a QuantizedTensor back to floating point.
    
    Args:
        qtensor: QuantizedTensor to dequantize
        
    Returns:
        Dequantized tensor in original shape and float32
    """
    config = qtensor.config
    n_elements = np.prod(qtensor.original_shape)
    n_groups = len(qtensor.scales)
    
    dequantized = np.zeros(n_elements, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * config.group_size
        end = min(start + config.group_size, n_elements)
        scale = float(qtensor.scales[g])
        group_len = end - start
        
        if config.target_bits == QuantizationTarget.INT4:
            # Unpack INT4 from uint8 storage
            packed = qtensor.data
            # Calculate offset in packed array
            elements_before = g * config.group_size
            bytes_before = (elements_before + 1) // 2
            group_packed = packed[bytes_before:bytes_before + (group_len + 1) // 2]
            quantized_values = _unpack_int4(group_packed, group_len)
            dequantized[start:end] = quantized_values.astype(np.float32) * scale
        else:
            dequantized[start:end] = qtensor.data[start:end].astype(np.float32) * scale
    
    return dequantized.reshape(qtensor.original_shape).astype(np.float32)


def quantize_int8(tensor: np.ndarray, group_size: int = 128) -> QuantizedTensor:
    """Convenience function for INT8 quantization."""
    config = QuantizationConfig(target_bits=QuantizationTarget.INT8, group_size=group_size)
    return quantize_groupwise(tensor, config)


def quantize_int4(tensor: np.ndarray, group_size: int = 128) -> QuantizedTensor:
    """Convenience function for INT4 quantization."""
    config = QuantizationConfig(target_bits=QuantizationTarget.INT4, group_size=group_size)
    return quantize_groupwise(tensor, config)


def dequantize_int8(qtensor: QuantizedTensor) -> np.ndarray:
    """Convenience function for INT8 dequantization."""
    if qtensor.config.target_bits != QuantizationTarget.INT8:
        raise ValueError("QuantizedTensor is not INT8")
    return dequantize_groupwise(qtensor)


def dequantize_int4(qtensor: QuantizedTensor) -> np.ndarray:
    """Convenience function for INT4 dequantization."""
    if qtensor.config.target_bits != QuantizationTarget.INT4:
        raise ValueError("QuantizedTensor is not INT4")
    return dequantize_groupwise(qtensor)