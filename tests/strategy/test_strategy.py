"""
Tests for LDMARK Strategy Engine.

All tests are deterministic and do NOT depend on the actual developer machine.
Uses synthetic ModelAnalysis and HardwareProfile exclusively.
"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

import pytest
from src.ldmark.analysis.models import ArchitectureInfo, DType, ModelAnalysis, ParameterCounts, TensorInfo
from src.ldmark.hardware.profile import (
    CPUArchitecture, CPUProfile, HardwareProfile, GPURuntime, RuntimeProfile, SystemProfile
)
from src.ldmark.strategy.models import (
    Backend, BackendCompatibility, CompilationStrategy, ConstraintConfig,
    FeasibilityState, HardwareContext, ModelContext, RuntimeMemoryFeasibility,
    StorageFeasibility, StrategySet, WeightRepresentation,
)
from src.ldmark.strategy.feasibility import (
    calculate_actual_storage_bytes,
    calculate_effective_bits_per_weight,
    calculate_runtime_memory,
    calculate_storage_bytes,
    build_strategy,
    evaluate_backend_compatibility,
    evaluate_runtime_feasibility,
    evaluate_storage_feasibility,
    get_available_representations,
    KNOWN_FORMATS,
    BACKEND_KERNEL_AVAILABILITY,
)
from src.ldmark.strategy.engine import StrategyEngine, evaluate_strategies
from src.ldmark.strategy.scenarios import (
    create_synthetic_model_analysis,
    create_synthetic_hardware_profile,
    run_scenario,
    run_all_scenarios,
    analyze_27b_q1_g128,
    verify_internal_consistency,
    SCENARIO_A,
    SCENARIO_B,
    SCENARIO_C,
    SCENARIO_D,
    SCENARIO_E,
    SCENARIO_27B_Q1_G128,
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
        model_id="test-model",
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


class TestEffectiveBitsPerWeight:
    def test_int8_groupwise(self):
        bpw = calculate_effective_bits_per_weight(8.0, 128, 16)
        assert abs(bpw - 8.125) < 1e-9

    def test_int4_groupwise(self):
        bpw = calculate_effective_bits_per_weight(4.0, 128, 16)
        assert abs(bpw - 4.125) < 1e-9

    def test_binary_groupwise(self):
        bpw = calculate_effective_bits_per_weight(1.0, 128, 16)
        assert abs(bpw - 1.125) < 1e-9

    def test_ternary_groupwise(self):
        bpw = calculate_effective_bits_per_weight(math.log2(3), 128, 16)
        expected = (128 * math.log2(3) + 16) / 128
        assert abs(bpw - expected) < 1e-9

    def test_no_grouping(self):
        bpw = calculate_effective_bits_per_weight(8.0, None, 16)
        assert bpw == 8.0
        
        bpw = calculate_effective_bits_per_weight(8.0, 0, 16)
        assert bpw == 8.0

    def test_different_group_sizes(self):
        bpw_32 = calculate_effective_bits_per_weight(4.0, 32, 16)
        bpw_128 = calculate_effective_bits_per_weight(4.0, 128, 16)
        assert bpw_32 > bpw_128  # Smaller groups = more overhead


class TestStorageCalculations:
    def test_storage_calculation(self):
        storage = calculate_storage_bytes(1_000_000, 8.125)
        expected = math.ceil(1_000_000 * 8.125 / 8)
        assert storage == expected

    def test_storage_calculation_fp16(self):
        storage = calculate_storage_bytes(1_000_000, 16.0)
        assert storage == 2_000_000

    def test_actual_vs_theoretical_storage_no_grouping(self):
        theoretical = calculate_storage_bytes(1_000_000, 16.0)
        actual = calculate_actual_storage_bytes(1_000_000, 16.0, None, 0)
        assert actual == theoretical

    def test_actual_vs_theoretical_storage_grouped(self):
        theoretical = calculate_storage_bytes(1_000_000, 4.125)
        actual = calculate_actual_storage_bytes(1_000_000, 4.125, 128, 16)
        assert actual > theoretical  # Includes scale factors

    def test_q1_g128_storage(self):
        param_count = 27_000_000_000
        weight_bits = 1
        group_size = 128
        scale_bits = 16
        effective_bpw = calculate_effective_bits_per_weight(weight_bits, group_size, scale_bits)
        
        theoretical = calculate_storage_bytes(param_count, effective_bpw)
        actual = calculate_actual_storage_bytes(param_count, weight_bits, group_size, scale_bits)
        
        theoretical_gb = theoretical / (1024 ** 3)
        actual_gb = actual / (1024 ** 3)
        
        # Theoretical: 27B * 1.125 / 8 = 3.54 GB
        assert 3.5 < theoretical_gb < 3.6
        # Actual serialized = weight bits + scale factors = same as theoretical
        # because effective_bpw already includes scale overhead
        assert 3.5 < actual_gb < 3.6
        assert actual == theoretical  # When using effective_bpw, theoretical = actual


class TestRuntimeMemoryEstimation:
    def test_runtime_memory_no_architecture(self):
        result = calculate_runtime_memory(1_000_000, 8.0, None, 2048, 1)
        assert result["weights"] == 1_000_000
        assert result["kv_cache"] == 0
        assert result["activations"] == 0
        assert result["overhead"] == 100_000
        assert result["total"] == 1_100_000

    def test_runtime_memory_with_architecture(self):
        arch = ModelContext(
            model_id="test",
            parameter_count=1_000_000_000,
            architecture_name="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            max_context_length=4096,
            attention_type=None,
        )
        result = calculate_runtime_memory(1_000_000_000, 8.0, arch, 2048, 1)
        assert result["weights"] == 1_000_000_000
        assert result["kv_cache"] > 0
        assert result["activations"] > 0
        assert result["overhead"] > 0
        assert result["total"] > result["weights"]

    def test_runtime_memory_components_sum(self):
        arch = ModelContext(
            model_id="test",
            parameter_count=100_000,
            architecture_name="test",
            num_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            num_kv_heads=8,
            intermediate_size=2048,
            vocab_size=1000,
            max_context_length=512,
            attention_type=None,
        )
        result = calculate_runtime_memory(100_000, 4.0, arch, 512, 1)
        total = result["weights"] + result["kv_cache"] + result["activations"] + result["overhead"]
        assert result["total"] == total

    def test_kv_cache_scales_with_context(self):
        arch = ModelContext(
            model_id="test",
            parameter_count=1_000_000_000,
            architecture_name="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            max_context_length=100000,
            attention_type=None,
        )
        kv_4k = calculate_runtime_memory(1_000_000_000, 4.0, arch, 4096, 1)["kv_cache"]
        kv_16k = calculate_runtime_memory(1_000_000_000, 4.0, arch, 16384, 1)["kv_cache"]
        kv_32k = calculate_runtime_memory(1_000_000_000, 4.0, arch, 32768, 1)["kv_cache"]
        
        assert kv_16k == kv_4k * 4
        assert kv_32k == kv_4k * 8

    def test_kv_cache_gqa_reduction(self):
        arch_gqa = ModelContext(
            model_id="test",
            parameter_count=1_000_000_000,
            architecture_name="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            intermediate_size=11008,
            vocab_size=32000,
            max_context_length=4096,
            attention_type="gqa",
        )
        arch_mha = ModelContext(
            model_id="test",
            parameter_count=1_000_000_000,
            architecture_name="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
            intermediate_size=11008,
            vocab_size=32000,
            max_context_length=4096,
            attention_type=None,
        )
        
        kv_gqa = calculate_runtime_memory(1_000_000_000, 4.0, arch_gqa, 4096, 1)["kv_cache"]
        kv_mha = calculate_runtime_memory(1_000_000_000, 4.0, arch_mha, 4096, 1)["kv_cache"]
        
        assert kv_gqa == kv_mha // 4


class TestStorageFeasibility:
    def test_feasible_storage(self):
        result = evaluate_storage_feasibility(1_000_000_000, 2_000_000_000)
        assert result.state == FeasibilityState.FEASIBLE
        assert "satisfied" in result.explanation.lower()

    def test_infeasible_storage(self):
        result = evaluate_storage_feasibility(3_000_000_000, 2_000_000_000)
        assert result.state == FeasibilityState.INFEASIBLE
        assert "violated" in result.explanation.lower()

    def test_unknown_storage(self):
        result = evaluate_storage_feasibility(1_000_000_000, None)
        assert result.state == FeasibilityState.UNKNOWN
        assert "no storage constraint" in result.explanation.lower()

    def test_storage_with_actual_file_size(self):
        result = evaluate_storage_feasibility(
            1_000_000_000, 
            2_000_000_000,
            actual_file_size_bytes=1_500_000_000
        )
        assert result.state == FeasibilityState.FEASIBLE
        assert "actual_file_size_bytes" in result.assumptions


class TestRuntimeFeasibility:
    def test_feasible_runtime(self):
        runtime = {"weights": 1_000_000_000, "kv_cache": 500_000_000, "activations": 300_000_000, "overhead": 180_000_000, "total": 1_980_000_000}
        result = evaluate_runtime_feasibility(runtime, 3_000_000_000, 4_000_000_000, 2048, 1)
        assert result.state == FeasibilityState.FEASIBLE

    def test_infeasible_runtime_constraint(self):
        runtime = {"weights": 1_000_000_000, "kv_cache": 500_000_000, "activations": 300_000_000, "overhead": 180_000_000, "total": 1_980_000_000}
        result = evaluate_runtime_feasibility(runtime, 1_500_000_000, 4_000_000_000, 2048, 1)
        assert result.state == FeasibilityState.INFEASIBLE

    def test_infeasible_memory_pressure(self):
        runtime = {"weights": 1_000_000_000, "kv_cache": 500_000_000, "activations": 300_000_000, "overhead": 180_000_000, "total": 1_980_000_000}
        result = evaluate_runtime_feasibility(runtime, None, 1_500_000_000, 2048, 1)
        assert result.state == FeasibilityState.INFEASIBLE
        assert result.memory_pressure_ratio > 1.0

    def test_high_memory_pressure_warning(self):
        runtime = {"weights": 1_000_000_000, "kv_cache": 500_000_000, "activations": 300_000_000, "overhead": 180_000_000, "total": 1_980_000_000}
        result = evaluate_runtime_feasibility(runtime, None, 2_100_000_000, 2048, 1)
        assert result.state == FeasibilityState.FEASIBLE
        assert result.memory_pressure_ratio > 0.9
        assert "90%" in result.explanation


class TestBackendCompatibility:
    def test_fp16_cpu_compatible(self):
        results = evaluate_backend_compatibility(WeightRepresentation.FP16, [Backend.CPU], [Backend.CPU])
        assert len(results) == 1
        assert results[0].backend == Backend.CPU
        assert results[0].state == FeasibilityState.FEASIBLE
        assert results[0].has_kernel is True

    def test_binary_cpu_incompatible(self):
        results = evaluate_backend_compatibility(WeightRepresentation.BINARY, [Backend.CPU], [Backend.CPU])
        assert len(results) == 1
        assert results[0].state == FeasibilityState.INFEASIBLE
        assert results[0].has_kernel is False
        assert "NO known kernel support" in results[0].explanation

    def test_ldmark_formats_no_kernels(self):
        for fmt in [WeightRepresentation.LDMARK_BINARY, WeightRepresentation.LDMARK_TERNARY, WeightRepresentation.LDMARK_INT4, WeightRepresentation.LDMARK_INT8]:
            results = evaluate_backend_compatibility(fmt, [Backend.CPU, Backend.CUDA], [Backend.CPU, Backend.CUDA])
            for r in results:
                assert r.has_kernel is False
                assert r.state == FeasibilityState.INFEASIBLE

    def test_target_backends_filter(self):
        results = evaluate_backend_compatibility(WeightRepresentation.INT4, [Backend.CUDA], [Backend.CPU, Backend.CUDA])
        assert len(results) == 1
        assert results[0].backend == Backend.CUDA


class TestStrategyBuilding:
    def test_build_fp16_strategy(self):
        model = ModelContext(
            model_id="test", parameter_count=1_000_000_000,
            architecture_name="llama", num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_kv_heads=32, intermediate_size=11008,
            vocab_size=32000, max_context_length=4096, attention_type=None
        )
        hardware = HardwareContext(
            total_ram_bytes=16 * 1024 ** 3,
            available_ram_bytes=12 * 1024 ** 3,
            gpu_vram_bytes=None,
            supported_backends=[Backend.CPU],
            cpu_architecture="x86_64",
        )
        constraints = ConstraintConfig()
        
        strategy = build_strategy("fp16_0", WeightRepresentation.FP16, model, hardware, constraints, 4096, 1)
        
        assert strategy.strategy_id == "fp16_0"
        assert strategy.weight_representation == WeightRepresentation.FP16
        assert strategy.bits_per_weight == 16.0
        assert strategy.group_size is None
        # No storage constraint = UNKNOWN state
        assert strategy.storage_feasibility.state == FeasibilityState.UNKNOWN
        assert strategy.is_runtime_feasible

    def test_build_int4_strategy(self):
        model = ModelContext(
            model_id="test", parameter_count=1_000_000_000,
            architecture_name="llama", num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_kv_heads=32, intermediate_size=11008,
            vocab_size=32000, max_context_length=4096, attention_type=None
        )
        hardware = HardwareContext(
            total_ram_bytes=16 * 1024 ** 3,
            available_ram_bytes=12 * 1024 ** 3,
            gpu_vram_bytes=None,
            supported_backends=[Backend.CPU],
            cpu_architecture="x86_64",
        )
        constraints = ConstraintConfig()
        
        strategy = build_strategy("int4_0", WeightRepresentation.INT4, model, hardware, constraints, 4096, 1)
        
        assert strategy.weight_representation == WeightRepresentation.INT4
        assert abs(strategy.bits_per_weight - 4.125) < 1e-6
        assert strategy.group_size == 128
        assert strategy.scale_dtype == "fp16"
        assert strategy.scale_bits == 16

    def test_strategy_includes_warnings(self):
        model = ModelContext(
            model_id="test", parameter_count=10_000_000_000,
            architecture_name="llama", num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_kv_heads=32, intermediate_size=11008,
            vocab_size=32000, max_context_length=4096, attention_type=None
        )
        hardware = HardwareContext(
            total_ram_bytes=4 * 1024 ** 3,
            available_ram_bytes=2 * 1024 ** 3,
            gpu_vram_bytes=None,
            supported_backends=[Backend.CPU],
            cpu_architecture="x86_64",
        )
        constraints = ConstraintConfig(max_model_storage_bytes=1_000_000_000)
        
        strategy = build_strategy("fp16_0", WeightRepresentation.FP16, model, hardware, constraints, 4096, 1)
        
        assert not strategy.is_storage_feasible
        assert len(strategy.warnings) > 0
        assert any("exceeds" in w for w in strategy.warnings)


class TestStrategyEngine:
    def test_engine_generates_all_formats(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        engine = StrategyEngine()
        result = engine.evaluate(analysis, hardware)
        
        assert len(result.strategies) >= 13
        methods = [s.weight_representation.value for s in result.strategies]
        assert "fp16" in methods
        assert "int8" in methods
        assert "int4" in methods
        assert "binary" in methods
        assert "ternary" in methods
        assert "ldmark_binary" in methods
        assert "ldmark_int4" in methods

    def test_engine_without_ldmark_formats(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        engine = StrategyEngine(include_ldmark_formats=False)
        result = engine.evaluate(analysis, hardware)
        
        methods = [s.weight_representation.value for s in result.strategies]
        assert "ldmark_binary" not in methods
        assert "ldmark_int4" not in methods

    def test_engine_with_constraints(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(max_model_storage_bytes=100_000_000)
        engine = StrategyEngine()
        result = engine.evaluate(analysis, hardware, constraints)
        
        for s in result.strategies:
            if s.storage_feasibility.estimated_storage_bytes > 100_000_000:
                assert not s.is_storage_feasible

    def test_engine_with_runtime_constraint(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(max_runtime_memory_bytes=500_000_000)
        engine = StrategyEngine()
        result = engine.evaluate(analysis, hardware, constraints)
        
        for s in result.strategies:
            if s.runtime_feasibility.estimated_total_bytes > 500_000_000:
                assert not s.is_runtime_feasible

    def test_engine_impossible_constraints(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile()
        constraints = ConstraintConfig(max_model_storage_bytes=1, max_runtime_memory_bytes=1)
        engine = StrategyEngine()
        result = engine.evaluate(analysis, hardware, constraints)
        
        feasible = [s for s in result.strategies if s.is_fully_feasible]
        assert len(feasible) == 0
        assert any("No strategies are feasible" in n for n in result.notes)

    def test_engine_context_lengths(self):
        analysis = make_model_analysis(
            param_count=1_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=32,
        )
        hardware = make_hardware_profile(ram_gb=32, available_gb=24)
        engine = StrategyEngine()
        
        results = engine.evaluate_at_context_lengths(
            analysis, hardware, context_lengths=[4096, 16384, 32768]
        )
        
        assert len(results) == 3
        for ctx_len, result in results.items():
            assert result.context_length == ctx_len
        
        fp16_4k = next(s for s in results[4096].strategies if s.weight_representation.value == "fp16")
        fp16_16k = next(s for s in results[16384].strategies if s.weight_representation.value == "fp16")
        fp16_32k = next(s for s in results[32768].strategies if s.weight_representation.value == "fp16")
        
        assert fp16_16k.runtime_feasibility.estimated_kv_cache_bytes > fp16_4k.runtime_feasibility.estimated_kv_cache_bytes
        assert fp16_32k.runtime_feasibility.estimated_kv_cache_bytes > fp16_16k.runtime_feasibility.estimated_kv_cache_bytes


class TestStrategySet:
    def test_feasible_strategies_property(self):
        analysis = make_model_analysis(param_count=1_000_000_000)
        hardware = make_hardware_profile(ram_gb=16, available_gb=12)
        constraints = ConstraintConfig(max_model_storage_bytes=2_000_000_000)
        engine = StrategyEngine()
        result = engine.evaluate(analysis, hardware, constraints)
        
        feasible = result.feasible_strategies
        assert all(s.is_fully_feasible for s in feasible)
        
        infeasible = result.infeasible_strategies
        assert all(not s.is_fully_feasible for s in infeasible)

    def test_to_dict_and_json(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        engine = StrategyEngine()
        result = engine.evaluate(analysis, hardware)
        
        d = result.to_dict()
        assert "model_id" in d
        assert "strategies" in d
        assert "constraints_used" in d
        
        json_str = result.to_json()
        assert "model_id" in json_str


class TestConvenienceFunction:
    def test_evaluate_strategies_function(self):
        analysis = make_model_analysis()
        hardware = make_hardware_profile()
        result = evaluate_strategies(analysis, hardware)
        assert isinstance(result, StrategySet)
        assert len(result.strategies) >= 13


class TestScenarios:
    def test_scenario_a_runs(self):
        result = run_scenario(SCENARIO_A)
        assert isinstance(result, StrategySet)
        assert result.parameter_count == 7_000_000_000

    def test_scenario_b_runs(self):
        result = run_scenario(SCENARIO_B)
        assert result.parameter_count == 13_000_000_000

    def test_scenario_c_runs(self):
        result = run_scenario(SCENARIO_C)
        assert result.parameter_count == 27_000_000_000

    def test_scenario_d_runs(self):
        result = run_scenario(SCENARIO_D)
        assert result.parameter_count == 27_000_000_000

    def test_scenario_e_runs(self):
        result = run_scenario(SCENARIO_E)
        assert result.parameter_count == 27_000_000_000
        assert any(s.is_storage_feasible for s in result.strategies)

    def test_analyze_27b_q1_g128(self):
        analysis = analyze_27b_q1_g128()
        
        assert analysis["model_parameters"] == 27_000_000_000
        assert analysis["effective_bits_per_weight"] == 1.125
        assert "theoretical_representation" in analysis
        assert "actual_serialized_file" in analysis
        assert "runtime_memory_4k_context" in analysis
        assert "key_distinctions" in analysis
        
        theoretical_gb = analysis["theoretical_representation"]["gb"]
        actual_gb = analysis["actual_serialized_file"]["gb"]
        runtime_gb = analysis["runtime_memory_4k_context"]["total_gb"]
        
        # Theoretical: 27B * 1.125 / 8 = 3.54 GB
        assert 3.5 < theoretical_gb < 3.6
        # Actual serialized includes scale factor overhead
        assert 3.5 < actual_gb < 4.1
        # Runtime memory includes weights + KV cache + activations + overhead
        assert runtime_gb > actual_gb

    def test_verify_internal_consistency(self):
        result = verify_internal_consistency()
        assert result["consistent"] is True
        assert len(result["issues"]) == 0


class TestModelContext:
    def test_from_model_analysis(self):
        analysis = make_model_analysis(
            param_count=1_000_000_000,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
        )
        ctx = ModelContext.from_model_analysis(analysis)
        
        assert ctx.parameter_count == 1_000_000_000
        assert ctx.num_layers == 32
        assert ctx.hidden_size == 4096
        assert ctx.num_attention_heads == 32
        assert ctx.num_kv_heads == 8


class TestHardwareContext:
    def test_from_hardware_profile(self):
        hardware = make_hardware_profile(ram_gb=16, available_gb=12)
        ctx = HardwareContext.from_hardware_profile(hardware)
        
        assert ctx.total_ram_bytes == 16 * 1024 ** 3
        assert ctx.available_ram_bytes == 12 * 1024 ** 3
        assert ctx.supported_backends == [Backend.CPU]


class TestConstraintConfig:
    def test_to_dict(self):
        constraints = ConstraintConfig(
            max_model_storage_bytes=1_000_000_000,
            max_runtime_memory_bytes=2_000_000_000,
            preferred_precision="int4",
            preferred_backend="cuda",
            batch_size=2,
        )
        d = constraints.to_dict()
        assert d["max_model_storage_bytes"] == 1_000_000_000
        assert d["max_runtime_memory_bytes"] == 2_000_000_000
        assert d["preferred_precision"] == "int4"
        assert d["preferred_backend"] == "cuda"
        assert d["batch_size"] == 2

    def test_from_dict(self):
        d = {
            "max_model_storage_bytes": 1_000_000_000,
            "preferred_precision": "int8",
            "batch_size": 4,
        }
        constraints = ConstraintConfig.from_dict(d)
        assert constraints.max_model_storage_bytes == 1_000_000_000
        assert constraints.preferred_precision == "int8"
        assert constraints.batch_size == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])