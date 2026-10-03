"""
LDMARK Runtime - Dequantization

Reconstructs compressed tensors using artifact metadata.
Supports INT8, INT4, binary, ternary formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from .exceptions import DequantizationError, UnsupportedRuntimeFormat
from src.ldmark.artifact.format import TensorEncoding, TensorIndexEntry, QuantizationParams


@dataclass
class DequantizedTensor:
    """Result of tensor dequantization."""
    name: str
    data: np.ndarray
    original_shape: Tuple[int, ...]
    original_dtype: np.dtype
    compressed_size_bytes: int
    decompressed_size_bytes: int
    encoding: TensorEncoding
    quantization: Optional[QuantizationParams]
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.data.shape),
            "dtype": str(self.data.dtype),
            "original_shape": list(self.original_shape),
            "original_dtype": str(self.original_dtype),
            "compressed_size_bytes": self.compressed_size_bytes,
            "decompressed_size_bytes": self.decompressed_size_bytes,
            "encoding": self.encoding.value,
            "quantization": self.quantization.to_dict() if self.quantization else None,
        }


def dequantize_int8(
    data: np.ndarray,
    scales: np.ndarray,
    entry: TensorIndexEntry
) -> np.ndarray:
    """Dequantize INT8 group-wise quantized tensor."""
    if entry.quantization is None:
        raise DequantizationError(entry.name, "Missing quantization parameters")
    
    qparams = entry.quantization
    group_size = qparams.group_size
    target_bits = qparams.target_bits
    
    if target_bits != 8:
        raise DequantizationError(entry.name, f"Expected INT8, got {target_bits}-bit")
    
    # Data is already int8, scales are float16
    original_shape = tuple(entry.original_shape)
    n_elements = np.prod(original_shape)
    n_groups = len(scales)
    
    dequantized = np.zeros(n_elements, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_elements)
        scale = float(scales[g])
        group_len = end - start
        
        # Data is int8 stored as raw bytes
        group_data = data[start:end].astype(np.float32)
        dequantized[start:end] = group_data * scale
    
    return dequantized.reshape(original_shape)


def dequantize_int4(
    data: np.ndarray,
    scales: np.ndarray,
    entry: TensorIndexEntry
) -> np.ndarray:
    """Dequantize INT4 group-wise quantized tensor (packed)."""
    if entry.quantization is None:
        raise DequantizationError(entry.name, "Missing quantization parameters")
    
    qparams = entry.quantization
    group_size = qparams.group_size
    target_bits = qparams.target_bits
    
    if target_bits != 4:
        raise DequantizationError(entry.name, f"Expected INT4, got {target_bits}-bit")
    
    original_shape = tuple(entry.original_shape)
    n_elements = np.prod(original_shape)
    n_groups = len(scales)
    
    dequantized = np.zeros(n_elements, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_elements)
        scale = float(scales[g])
        group_len = end - start
        
        # Calculate offset in packed array
        elements_before = g * group_size
        bytes_before = (elements_before + 1) // 2
        group_packed = data[bytes_before:bytes_before + (group_len + 1) // 2]
        
        # Unpack INT4
        quantized_values = _unpack_int4(group_packed, group_len)
        dequantized[start:end] = quantized_values.astype(np.float32) * scale
    
    return dequantized.reshape(original_shape)


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


def dequantize_binary(
    data: np.ndarray,
    scales: np.ndarray,
    entry: TensorIndexEntry
) -> np.ndarray:
    """Dequantize binary (1-bit) tensor."""
    if entry.quantization is None:
        raise DequantizationError(entry.name, "Missing quantization parameters")
    
    qparams = entry.quantization
    group_size = qparams.group_size
    target_bits = qparams.target_bits
    
    if target_bits != 1:
        raise DequantizationError(entry.name, f"Expected binary (1-bit), got {target_bits}-bit")
    
    original_shape = tuple(entry.original_shape)
    n_weights = np.prod(original_shape)
    n_groups = len(scales)
    
    dequantized = np.zeros(n_weights, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_weights)
        scale = float(scales[g])
        
        for i in range(start, end):
            byte_idx = i // 8
            bit_idx = i % 8
            bit = (data[byte_idx] >> bit_idx) & 1
            # 0 -> -1, 1 -> +1 (sign encoding)
            dequantized[i] = scale * (1.0 if bit else -1.0)
    
    return dequantized.reshape(original_shape)


def dequantize_ternary(
    data: np.ndarray,
    scales: np.ndarray,
    entry: TensorIndexEntry
) -> np.ndarray:
    """Dequantize ternary (2-bit) tensor."""
    if entry.quantization is None:
        raise DequantizationError(entry.name, "Missing quantization parameters")
    
    qparams = entry.quantization
    group_size = qparams.group_size
    target_bits = qparams.target_bits
    
    # Ternary uses ~1.585 bits but stored as 2 bits
    if target_bits not in (1, 2):
        # Accept 1 as "ternary" in some configs
        pass
    
    original_shape = tuple(entry.original_shape)
    n_weights = np.prod(original_shape)
    n_groups = len(scales)
    
    dequantized = np.zeros(n_weights, dtype=np.float32)
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_weights)
        scale = float(scales[g])
        
        for i in range(start, end):
            bit_pos = i * 2
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            
            if bit_offset <= 6:
                trit = (data[byte_idx] >> bit_offset) & 0x3
            else:
                # Spans byte boundary
                trit = ((data[byte_idx] >> bit_offset) | 
                       (data[byte_idx + 1] << (8 - bit_offset))) & 0x3
            
            # 00=-1, 01=0, 10=+1
            if trit == 0:
                dequantized[i] = -scale
            elif trit == 1:
                dequantized[i] = 0.0
            else:  # trit == 2
                dequantized[i] = scale
    
    return dequantized.reshape(original_shape)


def dequantize_raw(
    data: np.ndarray,
    scales: Optional[np.ndarray],
    entry: TensorIndexEntry
) -> np.ndarray:
    """Dequantize raw (uncompressed) tensor - just convert dtype."""
    original_shape = tuple(entry.original_shape)
    original_dtype = np.dtype(entry.original_dtype)
    
    # Data should already be in the correct format, just reshape and convert
    return data.reshape(original_shape).astype(original_dtype)


def dequantize_tensor(
    data: np.ndarray,
    scales: Optional[np.ndarray],
    entry: TensorIndexEntry,
    target_dtype: np.dtype = np.float16
) -> DequantizedTensor:
    """
    Main dequantization entry point.
    
    Dispatches to format-specific dequantizer based on encoding.
    
    Args:
        data: Compressed tensor data (uint8 array from file)
        scales: Scale factors (float16 array) or None for raw
        entry: Tensor index entry with metadata
        target_dtype: Target dtype for output (default float16)
    
    Returns:
        DequantizedTensor with reconstructed data
    
    Raises:
        UnsupportedRuntimeFormat: If encoding is not supported
        DequantizationError: If dequantization fails
    """
    encoding = entry.encoding
    
    try:
        if encoding == TensorEncoding.RAW:
            result = dequantize_raw(data, scales, entry)
        elif encoding == TensorEncoding.GROUPWISE_QUANTIZED:
            if entry.quantization is None:
                raise DequantizationError(entry.name, "Missing quantization parameters")
            
            target_bits = entry.quantization.target_bits
            
            if target_bits == 8:
                result = dequantize_int8(data, scales, entry)
            elif target_bits == 4:
                result = dequantize_int4(data, scales, entry)
            elif target_bits == 1:
                result = dequantize_binary(data, scales, entry)
            else:
                raise UnsupportedRuntimeFormat(
                    f"groupwise_quantized_{target_bits}bit",
                    f"Unsupported target bits: {target_bits}"
                )
        elif encoding == TensorEncoding.BINARY_PACKED:
            result = dequantize_binary(data, scales, entry)
        elif encoding == TensorEncoding.TERNARY_PACKED:
            result = dequantize_ternary(data, scales, entry)
        else:
            raise UnsupportedRuntimeFormat(
                encoding.value,
                f"Encoding {encoding.value} not implemented in runtime"
            )
        
        # Convert to target dtype
        result = result.astype(target_dtype)
        
        compressed_bytes = entry.byte_length + entry.scale_byte_length
        decompressed_bytes = result.nbytes
        
        return DequantizedTensor(
            name=entry.name,
            data=result,
            original_shape=tuple(entry.original_shape),
            original_dtype=np.dtype(entry.original_dtype),
            compressed_size_bytes=compressed_bytes,
            decompressed_size_bytes=decompressed_bytes,
            encoding=encoding,
            quantization=entry.quantization,
        )
    
    except UnsupportedRuntimeFormat:
        raise
    except Exception as e:
        raise DequantizationError(entry.name, str(e)) from e


def validate_dequantization(
    original: np.ndarray,
    reconstructed: np.ndarray,
    tensor_name: str = ""
) -> dict:
    """
    Validate dequantization against original tensor.
    
    Returns error metrics.
    """
    from src.ldmark.compression.metrics import calculate_error_metrics
    
    # Ensure same shape
    if original.shape != reconstructed.shape:
        return {
            "error": f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
        }
    
    return calculate_error_metrics(original, reconstructed)