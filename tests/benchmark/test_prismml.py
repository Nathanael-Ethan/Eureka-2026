"""
Tests for LDMARK Benchmark Laboratory - PrismML
"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from src.ldmark.benchmark.prismml import (
    run_prismml_q1_0_g128_benchmark,
    run_prismml_scaling_benchmark,
    run_prismml_claim_validation,
    create_prismml_comparison_table,
    export_prismml_benchmark_report,
    run_full_prismml_validation_suite,
    PrismMLBenchmarkResult,
)
from src.ldmark.benchmark.model import ErrorDistribution
from src.ldmark.compression.prismml_calc import prismml_q1_0_g128_bits_per_weight


class TestPrismMLBenchmark:
    def test_q1_0_g128_bpw(self):
        """Test the theoretical bits per weight calculation."""
        bpw = prismml_q1_0_g128_bits_per_weight()
        assert bpw == 1.125
    
    def test_q1_0_g128_formula(self):
        """Test the explicit formula."""
        # (128 * 1 + 16) / 128 = 144 / 128 = 1.125
        bpw = (128 * 1 + 16) / 128
        assert bpw == 1.125
    
    def test_run_prismml_benchmark(self):
        """Run the full PrismML Q1_0_g128 validation benchmark."""
        tensor = np.random.default_rng(42).normal(0, 1, (256, 256)).astype(np.float32)
        
        result = run_prismml_q1_0_g128_benchmark(
            tensor=tensor,
            tensor_name="test_prismml",
            group_size=128,
            scale_dtype=np.float16,
            random_seed=42,
        )
        
        assert isinstance(result, PrismMLBenchmarkResult)
        assert result.theoretical_bpw == 1.125
        assert result.calculated_bpw > 1.0
        assert result.calculated_bpw < 1.2
        assert result.bpw_error >= 0
        assert result.storage_validation_passed is True
        assert result.mae > 0
        assert result.quantization_time_ms > 0
        assert result.dequantization_time_ms > 0
        assert "LDMARK representation experiment" in result.disclaimer
        assert result.ldmark_representation_experiment is True
        assert result.prismml_model_quality_claim is False
    
    def test_prismml_scaling_benchmark(self):
        """Test scaling across model sizes."""
        results = run_prismml_scaling_benchmark([
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000,
        ])
        
        assert results["bpw_constant_across_sizes"] is True
        assert results["theoretical_bpw"] == 1.125
        assert len(results["results"]) == 4
        
        # Check all have same bpw
        for r in results["results"]:
            assert abs(r["bits_per_weight"] - 1.125) < 1e-9
    
    def test_prismml_claim_validation(self):
        """Test PrismML 3.9 GB claim validation."""
        validation = run_prismml_claim_validation(
            claimed_gb=3.9,
            model_params=27_000_000_000,
            tolerance=0.1,
        )
        
        assert "claimed_gb" in validation
        assert "calculated_gb" in validation
        assert "difference_gb" in validation
        assert "relative_error_percent" in validation
        assert "within_tolerance" in validation
        assert "disclaimer" in validation
        
        # LDMARK calculates ~3.54 GB
        assert abs(validation["calculated_gb"] - 3.54) < 0.05
        assert validation["claimed_gb"] == 3.9
        assert validation["difference_gb"] > 0.3
    
    def test_prismml_comparison_table(self):
        """Test representation comparison table."""
        table = create_prismml_comparison_table()
        
        assert len(table) == 6  # FP32, FP16, INT8, INT4, Binary, Ternary
        
        # Check all entries have required fields
        for entry in table:
            assert "representation" in entry
            assert "bits_per_weight" in entry
            assert "total_gb" in entry
            assert "group_count" in entry
        
        # Find specific entries
        fp32 = next(e for e in table if e["representation"] == "FP32")
        int8 = next(e for e in table if "INT8" in e["representation"])
        binary = next(e for e in table if "Binary" in e["representation"])
        
        assert fp32["bits_per_weight"] == 32
        assert fp32["group_count"] == 0
        
        assert int8["bits_per_weight"] == (128 * 8 + 16) / 128  # 8.125
        assert int8["group_count"] > 0
        
        assert abs(binary["bits_per_weight"] - 1.125) < 1e-6
    
    def test_export_prismml_report(self):
        """Test exporting PrismML benchmark report."""
        tensor = np.random.default_rng(42).normal(0, 1, (128,)).astype(np.float32)
        result = run_prismml_q1_0_g128_benchmark(tensor, "test")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            export_prismml_benchmark_report(result, filepath)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert data["theoretical_bpw"] == 1.125
            assert "disclaimer" in data
            assert "LDMARK representation experiment" in data["disclaimer"]
        finally:
            os.unlink(filepath)
    
    def test_full_prismml_validation_suite(self):
        """Test the full validation suite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_results = run_full_prismml_validation_suite(
                tensor_shapes=[(128,), (256,), (64, 64)],
                output_dir=tmpdir,
            )
            
            assert "mathematical_validation" in suite_results
            assert "scaling_validation" in suite_results
            assert "claim_validation" in suite_results
            assert "representation_comparison" in suite_results
            assert "tensor_benchmarks" in suite_results
            assert "disclaimer" in suite_results
            
            assert len(suite_results["tensor_benchmarks"]) == 3
            
            # Check output files created
            assert os.path.exists(os.path.join(tmpdir, "prismml_full_validation.json"))
            assert os.path.exists(os.path.join(tmpdir, "prismml_q1_0_g128_(128,).json"))
            assert os.path.exists(os.path.join(tmpdir, "prismml_q1_0_g128_(256,).json"))
            assert os.path.exists(os.path.join(tmpdir, "prismml_q1_0_g128_(64, 64).json"))


class TestPrismMLBenchmarkResult:
    def test_to_dict(self):
        result = PrismMLBenchmarkResult(
            theoretical_bpw=1.125,
            calculated_bpw=1.125,
            bpw_error=0.0,
            bpw_relative_error=0.0,
            theoretical_storage_bytes=1000,
            actual_storage_bytes=1000,
            storage_overhead_ratio=1.0,
            storage_validation_passed=True,
            mae=0.01,
            mse=0.001,
            max_absolute_error=0.1,
            relative_error=0.05,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,100),
            quantization_time_ms=1.0,
            dequantization_time_ms=1.0,
            validation_time_ms=1.0,
        )
        
        d = result.to_dict()
        assert d["theoretical_bpw"] == 1.125
        assert d["ldmark_representation_experiment"] is True
        assert d["prismml_model_quality_claim"] is False
    
    def test_to_json(self):
        result = PrismMLBenchmarkResult(
            theoretical_bpw=1.125,
            calculated_bpw=1.125,
            bpw_error=0.0,
            bpw_relative_error=0.0,
            theoretical_storage_bytes=1000,
            actual_storage_bytes=1000,
            storage_overhead_ratio=1.0,
            storage_validation_passed=True,
            mae=0.01,
            mse=0.001,
            max_absolute_error=0.1,
            relative_error=0.05,
            error_distribution=ErrorDistribution(0,0,0,0,0,0,0,0,0,100),
            quantization_time_ms=1.0,
            dequantization_time_ms=1.0,
            validation_time_ms=1.0,
        )
        
        json_str = result.to_json()
        assert "1.125" in json_str
        assert "LDMARK representation experiment" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])