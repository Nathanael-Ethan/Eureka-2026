"""M2: kernels/ vs runtime/dequantize.py parity on real fixture slices.

Reuses kernels/int4.py, binary.py, ternary.py, compression/*, runtime/dequantize.py.
Parity must be exact (same unpack math) — max diff < 1e-6; direct-vs-reference
matmul max-abs < 1e-3 (existing threshold, Luke-approved).
"""
import numpy as np
import pytest

from ldmark.compression.quantize import quantize_int4, quantize_int8, dequantize_groupwise
from ldmark.compression.binary_ternary import (
    BinaryConfig, TernaryConfig, binary_quantize_sign, binary_dequantize,
    ternary_quantize_threshold, ternary_dequantize,
)
from ldmark.kernels import Int4Matrix, BinaryMatrix, TernaryMatrix
from ldmark.compute import ComputationConfig, ComputePrecision, compare_results
from ldmark.kernels import int4_matmul_direct, int4_matmul_reference
from ldmark.runtime.dequantize import (
    dequantize_int4 as rt_int4, dequantize_binary as rt_binary,
    dequantize_ternary as rt_ternary,
)
from ldmark.artifact.format import TensorIndexEntry, QuantizationParams, TensorEncoding
import os

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..",
                       "experiments", "real_model", "tiny_llama_2L_128H.npz")


def _fixture_matrix():
    d = np.load(FIXTURE)
    w = d["model.layers.0.mlp.gate_proj.weight"].astype(np.float32)  # (512, 128)
    return np.ascontiguousarray(w[:64, :128])  # (64, 128) horizontal-valid for g128/g32


def _entry(name, shape, bits, gs, encoding):
    n = int(np.prod(shape))
    itemsize = np.dtype(np.float16).itemsize if False else 4
    return TensorIndexEntry(
        name=name, original_shape=list(shape), original_dtype="float32",
        original_size_bytes=n * 4, encoding=encoding,
        quantization=QuantizationParams(target_bits=bits, group_size=gs,
                                        scale_dtype="float16", symmetric=True),
    )


def test_int4_kernel_vs_compression_parity():
    w = _fixture_matrix()
    for gs in (128, 32):
        qt = quantize_int4(w, group_size=gs)
        kd = Int4Matrix(qt, input_shape=w.shape).dequantize()
        cd = dequantize_groupwise(qt)
        assert np.max(np.abs(kd - cd)) < 1e-6


def test_int4_runtime_vs_compression_parity():
    w = _fixture_matrix()
    for gs in (128, 32):
        qt = quantize_int4(w, group_size=gs)
        cd = dequantize_groupwise(qt)
        rd = rt_int4(qt.data, qt.scales,
                     _entry("t", w.shape, 4, gs, TensorEncoding.GROUPWISE_QUANTIZED))
        assert np.max(np.abs(rd.astype(np.float32) - cd)) < 1e-6


def test_binary_kernel_runtime_parity():
    w = _fixture_matrix()
    bt = binary_quantize_sign(w, BinaryConfig(group_size=128))
    ref = binary_dequantize(bt)
    assert np.max(np.abs(BinaryMatrix(bt).dequantize() - ref)) < 1e-6
    rd = rt_binary(bt.packed_data, bt.scales,
                   _entry("t", w.shape, 1, 128, TensorEncoding.BINARY_PACKED))
    assert np.max(np.abs(rd.astype(np.float32) - ref)) < 1e-6


def test_ternary_kernel_runtime_parity():
    w = _fixture_matrix()
    tt = ternary_quantize_threshold(w, TernaryConfig(group_size=128))
    ref = ternary_dequantize(tt)
    assert np.max(np.abs(TernaryMatrix(tt).dequantize() - ref)) < 1e-6
    rd = rt_ternary(tt.packed_data, tt.scales,
                    _entry("t", w.shape, 2, 128, TensorEncoding.TERNARY_PACKED))
    assert np.max(np.abs(rd.astype(np.float32) - ref)) < 1e-6


def test_int4_direct_matmul_parity_fixture():
    w = _fixture_matrix()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (128, 8)).astype(np.float32)
    qt = quantize_int4(w, group_size=32)
    m = Int4Matrix(qt, input_shape=w.shape)
    cfg = ComputationConfig(output_precision=ComputePrecision.FP32, accumulate_in_fp32=True)
    comp = compare_results(int4_matmul_direct(m, x, cfg), int4_matmul_reference(m, x, cfg),
                           rtol=1e-4, atol=1e-5)
    assert comp["passed"], comp
    assert comp["max_absolute_error"] < 1e-3
