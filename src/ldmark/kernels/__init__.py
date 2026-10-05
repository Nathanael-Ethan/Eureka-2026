"""
LDMARK Low-Bit Kernels Package

Direct computation kernels for quantized weight representations.
"""

from __future__ import annotations

from .int8 import (
    Int8Matrix,
    Int8MatmulHorizontal,
    int8_matmul_direct,
    int8_matmul_reference,
    benchmark_int8,
)
from .int4 import (
    Int4Matrix,
    Int4MatmulHorizontal,
    int4_matmul_direct,
    int4_matmul_reference,
    benchmark_int4,
)
from .binary import (
    BinaryMatrix,
    BinaryMatmulHorizontal,
    binary_matmul_direct,
    binary_matmul_reference,
    benchmark_binary,
)
from .ternary import (
    TernaryMatrix,
    TernaryMatmulHorizontal,
    ternary_matmul_direct,
    ternary_matmul_reference,
    benchmark_ternary,
)

__all__ = [
    # INT8
    "Int8Matrix",
    "Int8MatmulHorizontal",
    "int8_matmul_direct",
    "int8_matmul_reference",
    "benchmark_int8",
    # INT4
    "Int4Matrix",
    "Int4MatmulHorizontal",
    "int4_matmul_direct",
    "int4_matmul_reference",
    "benchmark_int4",
    # Binary
    "BinaryMatrix",
    "BinaryMatmulHorizontal",
    "binary_matmul_direct",
    "binary_matmul_reference",
    "benchmark_binary",
    # Ternary
    "TernaryMatrix",
    "TernaryMatmulHorizontal",
    "ternary_matmul_direct",
    "ternary_matmul_reference",
    "benchmark_ternary",
]