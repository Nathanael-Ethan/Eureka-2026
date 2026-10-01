import pytest
from src.ldmark.analysis.compression import (
    QuantizationFormat,
    COMMON_FORMATS,
    get_format_by_name,
    estimate_compression_for_format,
    estimate_all_compressions,
    estimate_compression_from_parameter_counts,
    calculate_effective_bps,
    create_custom_format,
    get_compression_summary,
)
from src.ldmark.analysis.models import DType, ParameterCounts


class TestQuantizationFormat:
    def test_estimate(self):
        fmt = QuantizationFormat("TEST", 4.0, group_size=32, scale_dtype=DType.FP16)
        est = fmt.estimate(1_000_000_000, DType.FP16)
        assert est.format_name == "TEST"
        assert est.effective_bits_per_weight == 4.0
        assert est.group_size == 32
        assert est.scale_dtype == DType.FP16

    def test_estimate_with_group_scale(self):
        fmt = QuantizationFormat("TEST", 1.125, group_size=128, scale_dtype=DType.FP16)
        est = fmt.estimate(1_000_000_000, DType.FP16)
        scale_bytes = (1_000_000_000 + 127) // 128 * 2
        expected_bytes = int(1_000_000_000 * 1.125 / 8) + scale_bytes
        assert est.estimated_bytes == expected_bytes


class TestCommonFormats:
    def test_formats_exist(self):
        assert len(COMMON_FORMATS) > 10
        names = [f.name for f in COMMON_FORMATS]
        assert "FP16" in names
        assert "INT4" in names
        assert "Bonsai Q1_0_g128" in names
        assert "Q4_K_M" in names

    def test_bonsai_format(self):
        fmt = get_format_by_name("Bonsai Q1_0_g128")
        assert fmt is not None
        assert fmt.effective_bits_per_weight == 1.125
        assert fmt.group_size == 128
        assert fmt.scale_dtype == DType.FP16


class TestGetFormatByName:
    def test_exact_match(self):
        fmt = get_format_by_name("INT4")
        assert fmt is not None
        assert fmt.name == "INT4"

    def test_case_insensitive(self):
        fmt = get_format_by_name("int4")
        assert fmt is not None

    def test_not_found(self):
        fmt = get_format_by_name("NONEXISTENT")
        assert fmt is None


class TestEstimateCompressionForFormat:
    def test_valid_format(self):
        est = estimate_compression_for_format(1_000_000_000, "INT4", DType.FP16)
        assert est is not None
        assert est.format_name == "INT4"
        assert est.compression_ratio == 4.0

    def test_invalid_format(self):
        est = estimate_compression_for_format(1_000_000_000, "NONEXISTENT")
        assert est is None


class TestCalculateEffectiveBps:
    def test_no_group(self):
        bps = calculate_effective_bps(4, 1)
        assert bps == 4.0

    def test_with_group(self):
        bps = calculate_effective_bps(1, 128, 16)
        assert bps == pytest.approx(1.125, rel=0.001)

    def test_with_zero_point(self):
        bps = calculate_effective_bps(4, 32, 16, 16)
        assert bps == pytest.approx(5.0, rel=0.001)


class TestCreateCustomFormat:
    def test_basic(self):
        fmt = create_custom_format("CUSTOM_INT4", 4, group_size=32, scale_dtype=DType.FP16)
        assert fmt.name == "CUSTOM_INT4"
        assert fmt.effective_bits_per_weight == pytest.approx(4.5, rel=0.001)

    def test_no_group(self):
        fmt = create_custom_format("CUSTOM_FP8", 8)
        assert fmt.effective_bits_per_weight == 8.0


class TestGetCompressionSummary:
    def test_summary(self):
        estimates = estimate_all_compressions(1_000_000_000)
        summary = get_compression_summary(estimates)
        assert "formats" in summary
        assert len(summary["formats"]) == len(COMMON_FORMATS)
        assert summary["best_compression"] is not None
        assert summary["smallest_size_gb"] is not None

    def test_sorted_by_bps(self):
        estimates = estimate_all_compressions(1_000_000_000)
        summary = get_compression_summary(estimates)
        bps_values = [f["effective_bits_per_weight"] for f in summary["formats"]]
        assert bps_values == sorted(bps_values)