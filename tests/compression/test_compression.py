"""
Tests for LDMARK compression laboratory.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.ldmark.compression.quantize import (
    QuantizationConfig,
    QuantizationTarget,
    QuantizedTensor,
    calculate_scale,
    quantize_groupwise,
    dequantize_groupwise,
    quantize_int8,
    quantize_int4,
    dequantize_int8,
    dequantize_int4,
)
from src.ldmark.compression.metrics import (
    calculate_error_metrics,
    calculate_compression_ratio,
    calculate_storage_size,
    compute_full_metrics,
)
from src.ldmark.compression.binary_ternary import (
    BinaryConfig,
    TernaryConfig,
    binary_quantize_sign,
    binary_dequantize,
    ternary_quantize_threshold,
    ternary_dequantize,
    calculate_binary_storage,
    calculate_ternary_storage,
)
from src.ldmark.compression.prismml_calc import (
    PrismMLConfig,
    prismml_q1_0_g128_bits_per_weight,
    calculate_prismml_storage,
    calculate_generic_groupwise_storage,
    compare_representations,
    validate_prismml_claim,
)
from src.ldmark.compression.experiment import (
    ExperimentConfig,
    run_quantization_experiment,
)


# ---------- INT8 Quantization Tests ----------

class TestINT8Quantization:
    def test_quantize_int8_basic(self):
        tensor = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=4)
        assert qtensor.data.dtype == np.int8
        assert len(qtensor.data) == 4
        assert len(qtensor.scales) == 1

    def test_quantize_int8_scale_correct(self):
        # Max abs = 1.0, qmax = 127 -> scale = 1/127
        tensor = np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=4)
        expected_scale = np.float16(1.0 / 127.0)
        assert np.isclose(float(qtensor.scales[0]), float(expected_scale), rtol=1e-2)

    def test_quantize_int8_range(self):
        tensor = np.linspace(-1, 1, 1000, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=128)
        assert np.all(qtensor.data >= -128)
        assert np.all(qtensor.data <= 127)

    def test_quantize_int8_zeros(self):
        tensor = np.zeros(16, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=8)
        assert np.all(qtensor.data == 0)


# ---------- INT4 Quantization Tests ----------

class TestINT4Quantization:
    def test_quantize_int4_basic(self):
        tensor = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        qtensor = quantize_int4(tensor, group_size=4)
        # 4 values packed into 2 bytes
        assert len(qtensor.data) == 2
        assert len(qtensor.scales) == 1

    def test_quantize_int4_range(self):
        tensor = np.linspace(-1, 1, 1000, dtype=np.float32)
        qtensor = quantize_int4(tensor, group_size=128)
        # Packed bytes are uint8 [0, 255]
        assert np.all(qtensor.data >= 0)
        assert np.all(qtensor.data <= 255)
        # Verify dequantized values are in correct range
        deq = dequantize_groupwise(qtensor)
        assert np.all(deq >= -8)
        assert np.all(deq <= 7)

    def test_quantize_int4_fewer_levels_than_int8(self):
        tensor = np.linspace(-1, 1, 128, dtype=np.float32)
        q8 = quantize_int8(tensor, group_size=128)
        q4 = quantize_int4(tensor, group_size=128)
        unique8 = len(np.unique(q8.data))
        unique4 = len(np.unique(q4.data))
        assert unique4 <= unique8


# ---------- Dequantization Tests ----------

class TestDequantization:
    def test_dequantize_int8_shape(self):
        tensor = np.random.default_rng(0).normal(0, 1, (4, 8)).astype(np.float32)
        qtensor = quantize_int8(tensor, group_size=16)
        deq = dequantize_groupwise(qtensor)
        assert deq.shape == tensor.shape
        assert deq.dtype == np.float32

    def test_dequantize_int8_zeros(self):
        tensor = np.zeros(32, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=8)
        deq = dequantize_groupwise(qtensor)
        assert np.allclose(deq, 0.0)

    def test_dequantize_roundtrip_error_bounded(self):
        tensor = np.linspace(-1, 1, 256, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=128)
        deq = dequantize_groupwise(qtensor)
        # INT8 error should be small
        assert np.max(np.abs(tensor - deq)) < 0.02

    def test_dequantize_int4_roundtrip(self):
        tensor = np.linspace(-1, 1, 256, dtype=np.float32)
        qtensor = quantize_int4(tensor, group_size=128)
        deq = dequantize_groupwise(qtensor)
        assert deq.shape == tensor.shape
        assert np.max(np.abs(tensor - deq)) < 0.2

    def test_dequantize_int8_helper_rejects_int4(self):
        tensor = np.ones(16, dtype=np.float32)
        qtensor = quantize_int4(tensor, group_size=8)
        with pytest.raises(ValueError):
            dequantize_int8(qtensor)

    def test_dequantize_int4_helper_rejects_int8(self):
        tensor = np.ones(16, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=8)
        with pytest.raises(ValueError):
            dequantize_int4(qtensor)


# ---------- Scale Calculation Tests ----------

class TestScaleCalculation:
    def test_scale_symmetric(self):
        group = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        scale = calculate_scale(group, QuantizationTarget.INT8, symmetric=True)
        # max_abs = 2.0, qmax = 127
        assert np.isclose(float(scale), 2.0 / 127.0, rtol=1e-2)

    def test_scale_asymmetric(self):
        group = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        scale = calculate_scale(group, QuantizationTarget.INT8, symmetric=False)
        # (max - min) / (2^8 - 1) = 3.0 / 255
        assert np.isclose(float(scale), 3.0 / 255.0, rtol=1e-2)

    def test_scale_zero_group(self):
        group = np.zeros(8, dtype=np.float32)
        scale = calculate_scale(group, QuantizationTarget.INT8, symmetric=True)
        assert float(scale) == 1.0

    def test_scale_int4(self):
        group = np.array([7.0, -7.0], dtype=np.float32)
        scale = calculate_scale(group, QuantizationTarget.INT4, symmetric=True)
        # max_abs = 7, qmax = 7
        assert np.isclose(float(scale), 1.0, rtol=1e-2)


# ---------- Group Handling Tests ----------

class TestGroupHandling:
    def test_group_count_exact(self):
        tensor = np.ones(256, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=128)
        assert len(qtensor.scales) == 2

    def test_group_count_remainder(self):
        tensor = np.ones(300, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=128)
        # 300 / 128 = 2.34 -> 3 groups
        assert len(qtensor.scales) == 3

    def test_group_size_larger_than_tensor(self):
        tensor = np.ones(10, dtype=np.float32)
        qtensor = quantize_int8(tensor, group_size=128)
        assert len(qtensor.scales) == 1

    def test_non_divisible_tensor(self):
        tensor = np.arange(1, 101, dtype=np.float32).reshape(10, 10)
        qtensor = quantize_int8(tensor, group_size=7)
        assert np.prod(qtensor.original_shape) == 100
        deq = dequantize_groupwise(qtensor)
        assert deq.shape == (10, 10)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError):
            QuantizationConfig(group_size=0)


# ---------- Binary Representation Tests ----------

class TestBinaryRepresentation:
    def test_binary_storage_calculation(self):
        config = BinaryConfig(group_size=128, scale_dtype=np.float16)
        # 128 weights = 128 bits + 16 bits scale = 144 bits = 18 bytes
        size = calculate_binary_storage(128, config)
        assert size == 18

    def test_binary_bits_per_weight_formula(self):
        config = BinaryConfig(group_size=128, scale_dtype=np.float16)
        # 1 bit + 16/128 = 1.125
        bpw = 1.0 + config.scale_overhead_bits_per_weight()
        assert np.isclose(bpw, 1.125, rtol=1e-6)

    def test_binary_quantize_sign(self):
        tensor = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        config = BinaryConfig(group_size=4)
        btensor = binary_quantize_sign(tensor, config)
        deq = binary_dequantize(btensor)
        # Signs should match
        assert np.all(np.sign(tensor) == np.sign(deq))

    def test_binary_dequantize_shape(self):
        tensor = np.ones((2, 4), dtype=np.float32)
        config = BinaryConfig(group_size=4)
        btensor = binary_quantize_sign(tensor, config)
        deq = binary_dequantize(btensor)
        assert deq.shape == (2, 4)


# ---------- Ternary Representation Tests ----------

class TestTernaryRepresentation:
    def test_ternary_storage_calculation(self):
        config = TernaryConfig(group_size=128, scale_dtype=np.float16)
        size = calculate_ternary_storage(128, config)
        # 128 * log2(3) = 202.8 bits + 16 bits = 218.8 bits = 28 bytes
        assert size == int(np.ceil((128 * np.log2(3) + 16) / 8))

    def test_ternary_bits_per_weight(self):
        config = TernaryConfig(group_size=128, scale_dtype=np.float16)
        assert np.isclose(config.bits_per_weight(), np.log2(3), rtol=1e-6)

    def test_ternary_quantize_threshold(self):
        tensor = np.array([1.0, -1.0, 0.01, -0.01], dtype=np.float32)
        config = TernaryConfig(group_size=4)
        ttensor = ternary_quantize_threshold(tensor, config, threshold=0.05)
        deq = ternary_dequantize(ttensor)
        # 0.01 and -0.01 should be zeroed
        assert deq[2] == 0.0
        assert deq[3] == 0.0

    def test_ternary_roundtrip_shape(self):
        tensor = np.random.default_rng(1).normal(0, 1, 64).astype(np.float32)
        config = TernaryConfig(group_size=16)
        ttensor = ternary_quantize_threshold(tensor, config, threshold=0.1)
        deq = ternary_dequantize(ttensor)
        assert deq.shape == tensor.shape


# ---------- Compression Ratio Tests ----------

class TestCompressionRatio:
    def test_ratio_basic(self):
        assert calculate_compression_ratio(1000, 500) == 2.0

    def test_ratio_identity(self):
        assert calculate_compression_ratio(1000, 1000) == 1.0

    def test_ratio_zero_compressed(self):
        assert calculate_compression_ratio(1000, 0) == float('inf')

    def test_storage_size_calc(self):
        # 128 weights at 4 bits = 512 bits + 16 bit scale = 528 bits -> 66 bytes
        size = calculate_storage_size(128, 4.0, num_scales=1, scale_bits=16)
        assert size == 66

    def test_int8_compression_vs_fp32(self):
        tensor = np.random.default_rng(2).normal(0, 1, 1024).astype(np.float32)
        qtensor = quantize_int8(tensor, group_size=128)
        metrics = compute_full_metrics(
            tensor,
            dequantize_groupwise(qtensor),
            qtensor.data,
            qtensor.scales,
            8,
            128,
        )
        # FP32 (1024 * 4 bytes) vs INT8 + scales
        assert metrics.compression_ratio > 3.0
        assert metrics.compression_ratio < 5.0


# ---------- Error Metrics Tests ----------

class TestErrorMetrics:
    def test_mae_perfect(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mae, mse, max_err, rel_err = calculate_error_metrics(a, a)
        assert mae == 0.0
        assert mse == 0.0
        assert max_err == 0.0
        assert rel_err == 0.0

    def test_mae_known(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        mae, mse, max_err, rel_err = calculate_error_metrics(a, b)
        assert mae == 1.0
        assert mse == 1.0
        assert max_err == 1.0

    def test_mse_known(self):
        a = np.array([0.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        mae, mse, max_err, rel_err = calculate_error_metrics(a, b)
        assert mae == 1.5
        assert mse == 2.5
        assert max_err == 2.0

    def test_relative_error(self):
        a = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
        b = np.array([0.0, 0.0], dtype=np.float32)
        mae, mse, max_err, rel_err = calculate_error_metrics(a, b)
        # ||diff|| / ||a|| = 5 / 5 = 1
        assert np.isclose(rel_err, 1.0)

    def test_error_shape_mismatch(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError):
            calculate_error_metrics(a, b)


# ---------- PrismML Mathematics Tests ----------

class TestPrismMLMath:
    def test_q1_0_g128_bpw(self):
        bpw = prismml_q1_0_g128_bits_per_weight(128, 1, 16)
        assert np.isclose(bpw, 1.125, rtol=1e-9)

    def test_bpw_formula_explicit(self):
        # (128 + 16) / 128 = 1.125
        assert (128 + 16) / 128 == 1.125

    def test_prismml_storage_27b(self):
        calc = calculate_prismml_storage(PrismMLConfig(model_params=27_000_000_000))
        # 27e9 * 1.125 bits / 8 / 1024^3
        expected_gb = 27_000_000_000 * 1.125 / 8 / (1024 ** 3)
        assert np.isclose(calc["total_gb"], expected_gb, rtol=1e-9)

    def test_prismml_claim_validation(self):
        result = validate_prismml_claim(claimed_gb=3.9, model_params=27_000_000_000)
        # ~3.54 GB calculated vs 3.9 claimed, within 10%
        assert result["within_tolerance"] is True

    def test_group_count(self):
        calc = calculate_prismml_storage(PrismMLConfig(model_params=27_000_000_000))
        assert calc["group_count"] == 27_000_000_000 // 128

    def test_generic_storage(self):
        result = calculate_generic_groupwise_storage(1000, 4, 128, 16)
        expected_bpw = (128 * 4 + 16) / 128
        assert np.isclose(result["bits_per_weight"], expected_bpw)

    def test_bit_widths(self):
        config = PrismMLConfig()
        assert config.weight_bits == 1
        assert config.scale_bits == 16
        assert config.group_size == 128

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            PrismMLConfig(group_size=0)
        with pytest.raises(ValueError):
            PrismMLConfig(model_params=0)


# ---------- Experiment Reproducibility Tests ----------

class TestExperimentReproducibility:
    def test_same_seed_same_result(self):
        cfg = ExperimentConfig(
            experiment_name="repro",
            input_shape=(64, 64),
            input_dtype=np.float32,
            target_bits=QuantizationTarget.INT8,
            group_size=128,
            seed=123,
        )
        r1 = run_quantization_experiment(cfg)
        r2 = run_quantization_experiment(cfg)
        assert np.allclose(r1.dequantized_tensor, r2.dequantized_tensor)
        assert r1.error_metrics["mae"] == r2.error_metrics["mae"]

    def test_different_seed_different_result(self):
        cfg1 = ExperimentConfig(
            experiment_name="a", input_shape=(64, 64), input_dtype=np.float32,
            target_bits=QuantizationTarget.INT8, group_size=128, seed=1,
        )
        cfg2 = ExperimentConfig(
            experiment_name="b", input_shape=(64, 64), input_dtype=np.float32,
            target_bits=QuantizationTarget.INT8, group_size=128, seed=2,
        )
        r1 = run_quantization_experiment(cfg1)
        r2 = run_quantization_experiment(cfg2)
        assert not np.allclose(r1.dequantized_tensor, r2.dequantized_tensor)

    def test_metadata_fields_present(self):
        cfg = ExperimentConfig(
            experiment_name="meta",
            input_shape=(32, 32),
            input_dtype=np.float16,
            target_bits=QuantizationTarget.INT4,
            group_size=64,
            seed=7,
        )
        r = run_quantization_experiment(cfg)
        d = r.to_dict()
        assert "experiment_id" in d
        assert "timestamp" in d
        assert d["config"]["input_shape"] == (32, 32)
        assert d["config"]["target_bits"] == "INT4"
        assert d["config"]["group_size"] == 64
        assert "bits_per_weight" in d["metrics"]
        assert "compression_ratio" in d["metrics"]

    def test_custom_tensor(self):
        custom = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        cfg = ExperimentConfig(
            experiment_name="custom",
            input_shape=(2, 2),
            input_dtype=np.float32,
            target_bits=QuantizationTarget.INT8,
            group_size=2,
            tensor_generator="custom",
            custom_tensor=custom,
        )
        r = run_quantization_experiment(cfg)
        assert r.config.custom_tensor is not None

    def test_custom_tensor_required(self):
        with pytest.raises(ValueError):
            ExperimentConfig(
                experiment_name="bad",
                input_shape=(2, 2),
                input_dtype=np.float32,
                target_bits=QuantizationTarget.INT8,
                group_size=2,
                tensor_generator="custom",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])