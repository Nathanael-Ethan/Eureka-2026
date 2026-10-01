"""
Tests for LDMARK compression planner.

All tests are deterministic and do NOT depend on the actual developer machine.
Uses synthetic ModelAnalysis and HardwareProfile exclusively.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import math
import pytest
from src.ldmark.analysis.models import (
    ArchitectureInfo,
    DType,
    ModelAnalysis,
    ParameterCounts,
    TensorInfo,
    TensorTrainableStatus,
)
from src.ldmark.hardware.profile import (
    CPUArchitecture,
    CPUProfile,
    HardwareProfile,
    RuntimeProfile,
    SystemProfile,
)
from src.ldmark.planning.planner import (
    CompressionMethod,
    CompressionPlan,
    CompressionPlanner,
    CompressionStrategy,
    ConstraintConfig,
    PlanResult,
    plan_compression,
    _calculate_groupwise_bits_per_weight,
    _calculate_storage_bytes,
    _calculate_runtime_memory,
)


def make_model_analysis(
    param_count=1_000_000_000,
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_kv_heads=32,
    intermediate_size=11008,
    vocab_size=32000,
    max_context_length=4096,
):
    params = ParameterCounts(
        total=param_count,
        by_dtype={DType.FP32: param_count},
        by_tensor={"model.embed_tokens.weight": vocab_size * hidden_size},
        trainable=param_count,
        non_trainable=0,
    )
    tensors = [
        TensorInfo(
            name="model.embed_tokens.weight",
            shape=[vocab_size, hidden_size],
            dtype=DType.FP32,
            num_elements=vocab_size * hidden_size,
            estimated_raw_storage_bytes=vocab_size * hidden_size * 4,
        )
    ]
    arch = ArchitectureInfo(
        model_architecture="llama",
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
    )
    return ModelAnalysis(
        model_id="test-llama-7b",
        source_path="/test/model",
        parameter_counts=params,
        tensors=tensors,
        architecture=arch,
        storage_estimates=[],
        compression_estimates=[],
    )


def make_hardware_profile(ram_gb=32, available_gb=16):
    return HardwareProfile(
        cpu=CPUProfile(
            architecture=CPUArchitecture.X86_64,
            model="Test CPU",
            core_count=8,
            thread_count=16,
        ),
        gpu=None,
        system=SystemProfile(
            total_ram_bytes=ram_gb * 1024 ** 3,
            total_ram_gb=float(ram_gb),
            available_ram_bytes=available_gb * 1024 ** 3,
            available_ram_gb=float(available_gb),
            operating_system="Linux",
        ),
        runtime=RuntimeProfile(supported_backends=[]),
    )


class TestGroupwiseMath:
    def test_int8_bpw(self):
        bpw = _calculate_groupwise_bits_per_weight(8.0, 128, 16)
        assert abs(bpw - 8.125) < 1e-9

    def test_int4_bpw(self):
        bpw = _calculate_groupwise_bits_per_weight(4.0, 128, 16)
        assert abs(bpw - 4.125) < 1e-9

    def test_binary_bpw(self):
        bpw = _calculate_groupwise_bits_per_weight(1.0, 128, 16)
        assert abs(bpw - 1.125) < 1e-9

    def test_ternary_bpw(self):
        bpw = _calculate_groupwise_bits_per_weight(math.log2(3), 128, 16)
        expected = (128 * math.log2(3) + 16) / 128
        assert abs(bpw - expected) < 1e-9

    def test_no_grouping(self):
        bpw = _calculate_groupwise_bits_per_weight(8.0, 0, 16)
        assert bpw == 8.0

    def test_storage_calculation(self):
        storage = _calculate_storage_bytes(1_000_000, 8.125)
        expected = math.ceil(1_000_000 * 8.125 / 8)
        assert storage == expected

    def test_storage_calculation_fp16(self):
        storage = _calculate_storage_bytes(1_000_000, 16.0)
        assert storage == 2_000_000


class TestRuntimeMemoryEstimation:
    def test_runtime_memory_no_architecture(self):
        result = _calculate_runtime_memory(1_000_000, 8.0, None)
        assert result["weights"] == 1_000_000
        assert result["kv_cache"] == 0
        assert result["activations"] == 0
        assert result["overhead"] == 100_000
        assert result["total"] == 1_100_000

    def test_runtime_memory_with_architecture(self):
        arch = ArchitectureInfo(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
        )
        result = _calculate_runtime_memory(1_000_000_000, 8.0, arch, context_length=2048)
        assert result["weights"] == 1_000_000_000
        assert result["kv_cache"] > 0
        assert result["activations"] > 0
        assert result["overhead"] > 0
        assert result["total"] > result["weights"]

    def test_runtime_memory_components_sum(self):
        arch = ArchitectureInfo(num_layers=2, hidden_size=512, num_attention_heads=8, num_kv_heads=8)
        result = _calculate_runtime_memory(100_000, 4.0, arch, context_length=512)
        total = result["weights"] + result["kv_cache"] + result["activations"] + result["overhead"]
        assert result["total"] == total


class TestCompressionPlanner:
    def test_plan_generates_all_methods(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        assert result.success
        assert len(result.plan.strategies) == 5
        methods = [s.method for s in result.plan.strategies]
        assert "fp16" in methods
        assert "int8" in methods
        assert "int4" in methods
        assert "binary" in methods
        assert "ternary" in methods

    def test_plan_with_no_constraints(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        assert result.success
        for s in result.plan.strategies:
            assert s.fits_storage_budget
            assert s.fits_runtime_budget

    def test_plan_with_tight_storage_constraint(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(max_model_storage_bytes=100_000_000)
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware, constraints)
        assert result.success
        for s in result.plan.strategies:
            if s.estimated_storage_bytes > 100_000_000:
                assert not s.fits_storage_budget

    def test_plan_with_tight_runtime_constraint(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(max_runtime_memory_bytes=500_000_000)
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware, constraints)
        assert result.success
        for s in result.plan.strategies:
            if s.estimated_runtime_total_memory_bytes > 500_000_000:
                assert not s.fits_runtime_budget

    def test_plan_impossible_constraints(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(
            max_model_storage_bytes=1,
            max_runtime_memory_bytes=1,
        )
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware, constraints)
        assert result.success
        feasible = [s for s in result.plan.strategies if s.fits_storage_budget and s.fits_runtime_budget]
        assert len(feasible) == 0
        assert any("No methods are feasible" in n for n in result.plan.notes)

    def test_plan_zero_parameters(self):
        analysis = make_model_analysis(param_count=0)
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        assert not result.success
        assert len(result.errors) > 0

    def test_strategy_has_warnings_when_over_budget(self):
        analysis = make_model_analysis(param_count=10_000_000_000)
        hardware = make_hardware_profile(ram_gb=4, available_gb=2)
        constraints = ConstraintConfig(max_model_storage_bytes=1_000_000_000)
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware, constraints)
        assert result.success
        fp16_strategy = next(s for s in result.plan.strategies if s.method == "fp16")
        assert not fp16_strategy.fits_storage_budget
        assert len(fp16_strategy.warnings) > 0

    def test_strategy_to_dict(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        for s in result.plan.strategies:
            d = s.to_dict()
            assert "method" in d
            assert "bits_per_weight" in d
            assert "estimated_storage_bytes" in d
            assert "fits_storage_budget" in d
            assert "fits_runtime_budget" in d
            assert "warnings" in d
            assert "assumptions" in d

    def test_plan_to_dict(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        d = result.plan.to_dict()
        assert "model_id" in d
        assert "parameter_count" in d
        assert "strategies" in d
        assert "constraints" in d
        assert "notes" in d

    def test_plan_to_json(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        json_str = result.plan.to_json()
        assert "model_id" in json_str
        assert "strategies" in json_str

    def test_compression_ratio_ordering(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        strategies = {s.method: s for s in result.plan.strategies}
        assert strategies["binary"].compression_ratio > strategies["int4"].compression_ratio
        assert strategies["int4"].compression_ratio > strategies["int8"].compression_ratio
        assert strategies["int8"].compression_ratio > strategies["fp16"].compression_ratio

    def test_bits_per_weight_values(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        strategies = {s.method: s for s in result.plan.strategies}
        assert strategies["fp16"].bits_per_weight == 16.0
        assert abs(strategies["int8"].bits_per_weight - 8.125) < 1e-6
        assert abs(strategies["int4"].bits_per_weight - 4.125) < 1e-6
        assert abs(strategies["binary"].bits_per_weight - 1.125) < 1e-6

    def test_27b_model_planning(self):
        analysis = make_model_analysis(
            param_count=27_000_000_000,
            num_layers=80,
            hidden_size=8192,
            num_attention_heads=64,
            num_kv_heads=8,
            intermediate_size=28672,
        )
        hardware = make_hardware_profile(ram_gb=16, available_gb=12)
        constraints = ConstraintConfig(
            max_model_storage_bytes=8 * 1024 ** 3,
            max_runtime_memory_bytes=12 * 1024 ** 3,
        )
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware, constraints)
        assert result.success
        strategies = {s.method: s for s in result.plan.strategies}
        assert not strategies["fp16"].fits_storage_budget
        assert not strategies["int8"].fits_storage_budget
        assert not strategies["int4"].fits_storage_budget
        assert strategies["binary"].fits_storage_budget


class TestConvenienceFunction:
    def test_plan_compression_function(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        result = plan_compression(analysis, hardware)
        assert result.success
        assert len(result.plan.strategies) == 5

    def test_plan_compression_with_constraints(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(max_model_storage_bytes=2_000_000_000)
        result = plan_compression(analysis, hardware, constraints)
        assert result.success


class TestConstraintConfig:
    def test_constraint_to_memory_budget(self):
        constraints = ConstraintConfig(
            max_model_storage_bytes=1000,
            max_runtime_memory_bytes=2000,
            minimum_free_memory_bytes=100,
        )
        budget = constraints.to_memory_budget()
        assert budget.max_model_storage_bytes == 1000
        assert budget.max_runtime_memory_bytes == 2000
        assert budget.minimum_free_memory_bytes == 100

    def test_constraint_to_dict(self):
        constraints = ConstraintConfig(
            max_model_storage_bytes=1000,
            target_precision="int8",
        )
        d = constraints.to_dict()
        assert d["max_model_storage_bytes"] == 1000
        assert d["target_precision"] == "int8"


class TestPlanResult:
    def test_plan_result_success(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        assert result.success
        assert isinstance(result.plan, CompressionPlan)
        assert len(result.errors) == 0

    def test_plan_result_failure(self):
        analysis = make_model_analysis(param_count=0)
        hardware = make_hardware_profile()
        planner = CompressionPlanner()
        result = planner.plan(analysis, hardware)
        assert not result.success
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])