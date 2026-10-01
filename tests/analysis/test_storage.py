import pytest
from src.ldmark.analysis.storage import (
    estimate_storage_for_dtype,
    estimate_all_storages,
    estimate_storage_from_parameter_counts,
    compare_storage_estimates,
    get_storage_summary,
    estimate_actual_vs_theoretical,
)
from src.ldmark.analysis.models import DType, ParameterCounts


class TestEstimateStorageForDtype:
    def test_fp16(self):
        est = estimate_storage_for_dtype(1_000_000_000, DType.FP16)
        assert est.dtype == DType.FP16
        assert est.estimated_bytes == 2_000_000_000
        assert est.estimated_gb == pytest.approx(1.8626, rel=0.01)

    def test_int4(self):
        est = estimate_storage_for_dtype(1_000_000_000, DType.INT4)
        assert est.estimated_bytes == 500_000_000
        assert est.estimated_gb == pytest.approx(0.4657, rel=0.01)

    def test_bit1(self):
        est = estimate_storage_for_dtype(1_000_000_000, DType.BIT1)
        assert est.estimated_bytes == 125_000_000


class TestEstimateAllStorages:
    def test_default_dtypes(self):
        estimates = estimate_all_storages(1_000_000_000)
        assert len(estimates) == 8
        dtypes = [e.dtype for e in estimates]
        assert DType.FP32 in dtypes
        assert DType.FP16 in dtypes
        assert DType.INT4 in dtypes
        assert DType.BIT1 in dtypes

    def test_custom_dtypes(self):
        estimates = estimate_all_storages(1_000_000_000, [DType.FP16, DType.INT8])
        assert len(estimates) == 2


class TestEstimateStorageFromParameterCounts:
    def test_with_by_dtype(self):
        counts = ParameterCounts()
        counts.add_tensor("w1", 1_000_000_000, DType.FP16, None)
        counts.add_tensor("w2", 500_000_000, DType.INT8, None)
        estimates = estimate_storage_from_parameter_counts(counts)
        assert len(estimates) == 8
        fp16_est = next(e for e in estimates if e.dtype == DType.FP16)
        assert fp16_est.estimated_bytes == 2_000_000_000

    def test_fallback_to_total(self):
        counts = ParameterCounts()
        counts.add_tensor("w1", 1_000_000_000, DType.FP16, None)
        estimates = estimate_storage_from_parameter_counts(counts, [DType.INT4])
        int4_est = estimates[0]
        assert int4_est.estimated_bytes == 1_000_000_000 * 0.5


class TestCompareStorageEstimates:
    def test_comparison(self):
        estimates = [
            estimate_storage_for_dtype(1_000_000_000, DType.FP16),
            estimate_storage_for_dtype(1_000_000_000, DType.INT4),
        ]
        comparison = compare_storage_estimates(estimates)
        assert comparison["baseline"] == "float16"
        assert "int4" in comparison["comparisons"]
        assert comparison["comparisons"]["int4"]["vs_baseline_ratio"] == 4.0


class TestEstimateActualVsTheoretical:
    def test_exact_match(self):
        result = estimate_actual_vs_theoretical(1_000_000, 1_000_000)
        assert result["ratio"] == 1.0
        assert result["overhead_percent"] == 0.0

    def test_larger_actual(self):
        result = estimate_actual_vs_theoretical(1_200_000, 1_000_000)
        assert result["ratio"] == 1.2
        assert result["overhead_percent"] == 20.0
        assert "metadata" in result["interpretation"].lower()

    def test_smaller_actual(self):
        result = estimate_actual_vs_theoretical(800_000, 1_000_000)
        assert result["ratio"] == 0.8
        assert "compressed" in result["interpretation"].lower()