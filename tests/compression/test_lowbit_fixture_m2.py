"""M2: low-bit roundtrip error bounds on real tiny_llama fixture.

Reuses compression/quantize.py, binary_ternary.py, prismml_calc.py, metrics.py.
Thresholds approved by Luke: INT8 MAE ~1.1e-4, INT4 MAE ~2e-3, binary ~9.6e-3,
ternary ~1.5e-2 (M1a/M1b locked). Asserts are upper bounds with headroom —
do NOT loosen without Luke sign-off.
"""
import os
import numpy as np
import pytest

from src.ldmark.compression.quantize import quantize_int8, quantize_int4, dequantize_groupwise
from src.ldmark.compression.binary_ternary import (
    BinaryConfig, TernaryConfig, binary_quantize_sign, binary_dequantize,
    ternary_quantize_threshold, ternary_dequantize,
)
from src.ldmark.compression.metrics import calculate_error_metrics
from src.ldmark.compression.prismml_calc import (
    prismml_q1_0_g128_bits_per_weight, calculate_prismml_storage, PrismMLConfig,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "..",
                       "experiments", "real_model", "tiny_llama_2L_128H.npz")


def _load_all():
    d = np.load(FIXTURE)
    return np.concatenate([d[k].ravel().astype(np.float32) for k in d.files])


def test_fixture_present():
    assert os.path.exists(FIXTURE), f"missing fixture {FIXTURE}"
    w = _load_all()
    assert w.size == 787072


def test_prismml_q1_0_g128_storage_math():
    assert prismml_q1_0_g128_bits_per_weight(128, 1, 16) == pytest.approx(1.125, rel=1e-9)
    calc = calculate_prismml_storage(PrismMLConfig(model_params=27_000_000_000))
    assert calc["bits_per_weight"] == pytest.approx(1.125, rel=1e-9)
    assert calc["group_count"] == 27_000_000_000 // 128


def test_int8_g128_error_bound_fixture():
    w = _load_all()
    deq = dequantize_groupwise(quantize_int8(w, group_size=128))
    mae, mse, mx, rel = calculate_error_metrics(w, deq)
    assert mae < 5e-4, mae
    assert mx < 2e-3, mx
    assert rel < 0.02, rel


def test_int4_g128_error_bound_fixture():
    w = _load_all()
    deq = dequantize_groupwise(quantize_int4(w, group_size=128))
    mae, mse, mx, rel = calculate_error_metrics(w, deq)
    assert mae < 5e-3, mae
    assert mx < 2e-2, mx
    assert rel < 0.15, rel


def test_int4_g32_beats_g128_fixture():
    w = _load_all()
    m32, _, _, _ = calculate_error_metrics(w, dequantize_groupwise(quantize_int4(w, group_size=32)))
    m128, _, _, _ = calculate_error_metrics(w, dequantize_groupwise(quantize_int4(w, group_size=128)))
    assert m32 <= m128


def test_binary_g128_q1_0_equivalent_bound_fixture():
    # PrismML Q1_0_g128 roundtrip proxy: 1-bit + FP16 scale per 128-group
    w = _load_all()
    deq = binary_dequantize(binary_quantize_sign(w, BinaryConfig(group_size=128)))
    mae, mse, mx, rel = calculate_error_metrics(w, deq)
    assert mae < 3e-2, mae
    assert rel < 0.7, rel


def test_ternary_g128_bound_fixture():
    w = _load_all()
    deq = ternary_dequantize(ternary_quantize_threshold(w, TernaryConfig(group_size=128)))
    mae, mse, mx, rel = calculate_error_metrics(w, deq)
    assert mae < 4e-2, mae
    assert rel < 0.7, rel
