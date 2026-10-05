"""
Tests for LDMARK Mixed-Precision Compilation Engine.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np
import pytest

from src.ldmark.mixed_precision.plan import (
    PrecisionType,
    TensorRole,
    TensorClassification,
    PrecisionCandidate,
    TensorPrecisionAssignment,
    MixedPrecisionPlan,
    create_mixed_precision_plan,
    create_uniform_plan,
    create_uniform_fp16_plan,
    get_precision_candidates,
)
from src.ldmark.mixed_precision.classification import (
    classify_tensor,
    TensorClassifier,
    TensorMetadata,
    classify_model_tensors,
)
from src.ldmark.mixed_precision.storage import (
    StorageComponent,
    StorageAccounting,
    MixedPrecisionStorageModel,
    calculate_mixed_precision_storage,
)
from src.ldmark.mixed_precision.error_analysis import (
    ErrorMeasurement,
    TensorErrorProfile,
    calculate_error_metrics,
)
from src.ldmark.mixed_precision.search import (
    SearchConstraint,
    Configuration,
    SearchResult,
    greedy_search,
)
from src.ldmark.mixed_precision.scenarios import (
    ModelScaleScenario,
    ScenarioResult,
    get_default_scenarios,
    estimate_parameter_distribution,
    create_classifications_from_scenario,
    create_mixed_precision_strategies,
)


# ---------- Plan Tests ----------

class TestPrecisionType:
    def test_bits_per_weight(self):
        assert PrecisionType.FP32.bits_per_weight == 32.0
        assert PrecisionType.FP16.bits_per_weight == 16.0
        assert PrecisionType.INT8.bits_per_weight == 8.0
        assert PrecisionType.INT4.bits_per_weight == 4.0
        assert PrecisionType.BINARY.bits_per_weight == 1.0
        assert abs(PrecisionType.TERNARY.bits_per_weight - 1.585) < 0.01
    
    def test_is_quantized(self):
        assert PrecisionType.INT8.is_quantized()
        assert PrecisionType.INT4.is_quantized()
        assert PrecisionType.BINARY.is_quantized()
        assert PrecisionType.TERNARY.is_quantized()
        assert not PrecisionType.FP32.is_quantized()
        assert not PrecisionType.FP16.is_quantized()


class TestTensorClassification:
    def test_creation(self):
        cls = TensorClassification(
            tensor_name="test",
            role=TensorRole.ATTENTION_Q,
            shape=(4096, 4096),
            num_parameters=4096 * 4096,
            dtype="float32",
            layer_index=0,
        )
        assert cls.role == TensorRole.ATTENTION_Q
        assert cls.num_parameters == 4096 * 4096
    
    def test_to_dict(self):
        cls = TensorClassification(
            tensor_name="test",
            role=TensorRole.MLP_UP,
            shape=(4096, 11008),
            num_parameters=4096 * 11008,
            dtype="float32",
        )
        d = cls.to_dict()
        assert d["role"] == "mlp_up"
        assert d["tensor_name"] == "test"


class TestPrecisionCandidate:
    def test_fp16_candidate(self):
        cand = PrecisionCandidate(precision=PrecisionType.FP16)
        assert cand.estimated_bits_per_weight == 16.0
    
    def test_int8_with_group(self):
        cand = PrecisionCandidate(precision=PrecisionType.INT8, group_size=128)
        # 8 + 16/128 = 8.125
        assert abs(cand.estimated_bits_per_weight - 8.125) < 0.001
    
    def test_int4_with_group(self):
        cand = PrecisionCandidate(precision=PrecisionType.INT4, group_size=128)
        # 4 + 16/128 = 4.125
        assert abs(cand.estimated_bits_per_weight - 4.125) < 0.001


class TestTensorPrecisionAssignment:
    def test_creation(self):
        cand = PrecisionCandidate(precision=PrecisionType.INT8, group_size=128)
        assignment = TensorPrecisionAssignment(
            tensor_name="test",
            precision=PrecisionType.INT8,
            group_size=128,
            candidate=cand,
        )
        assert assignment.tensor_name == "test"
        assert assignment.precision == PrecisionType.INT8
        assert assignment.key == "test"


class TestMixedPrecisionPlan:
    def test_create_uniform_plan(self):
        plan = create_uniform_plan("test", ["a", "b", "c"], PrecisionType.INT8, group_size=128)
        assert plan.total_tensors() == 3
        assert all(a.precision == PrecisionType.INT8 for a in plan.assignments.values())
        assert all(a.group_size == 128 for a in plan.assignments.values())
    
    def test_add_assignment(self):
        plan = MixedPrecisionPlan(model_identifier="test")
        new_plan = plan.add_assignment(TensorPrecisionAssignment(
            tensor_name="t1",
            precision=PrecisionType.INT8,
        ))
        assert new_plan.total_tensors() == 1
        assert plan.total_tensors() == 0  # Original unchanged
    
    def test_get_precision(self):
        plan = create_uniform_plan("test", ["a", "b"], PrecisionType.INT4)
        assert plan.get_precision("a") == PrecisionType.INT4
        assert plan.get_precision("c") is None
    
    def test_tensors_by_precision(self):
        plan = create_uniform_plan("test", ["a", "b", "c"], PrecisionType.INT8)
        plan = plan.add_assignment(TensorPrecisionAssignment(
            tensor_name="d", precision=PrecisionType.INT4
        ))
        int8_tensors = plan.tensors_by_precision(PrecisionType.INT8)
        int4_tensors = plan.tensors_by_precision(PrecisionType.INT4)
        assert len(int8_tensors) == 3
        assert len(int4_tensors) == 1
    
    def test_to_dict(self):
        plan = create_uniform_plan("test", ["a", "b"], PrecisionType.INT8)
        d = plan.to_dict()
        assert d["model_identifier"] == "test"
        assert d["total_tensors"] == 2
        assert d["precision_distribution"]["int8"] == 2


class TestPrecisionCandidates:
    def test_get_precision_candidates(self):
        cls = TensorClassification(
            tensor_name="test",
            role=TensorRole.ATTENTION_Q,
            shape=(4096, 4096),
            num_parameters=4096 * 4096,
            dtype="float32",
        )
        candidates = get_precision_candidates(cls)
        assert len(candidates) == 5  # FP16, INT8, INT4, BINARY, TERNARY
        
        # Check all have valid precisions
        for cand in candidates:
            assert cand.precision in [PrecisionType.FP16, PrecisionType.INT8, 
                                       PrecisionType.INT4, PrecisionType.BINARY, 
                                       PrecisionType.TERNARY]
            assert cand.estimated_bits_per_weight > 0


# ---------- Classification Tests ----------

class TestClassifyTensor:
    def test_embedding(self):
        cls = classify_tensor("model.embed_tokens.weight", (32000, 4096))
        assert cls.role == TensorRole.EMBEDDING
    
    def test_attention_q(self):
        cls = classify_tensor("layers.0.attn.q_proj.weight", (4096, 4096))
        assert cls.role == TensorRole.ATTENTION_Q
    
    def test_attention_k(self):
        cls = classify_tensor("layers.0.attn.k_proj.weight", (4096, 4096))
        assert cls.role == TensorRole.ATTENTION_K
    
    def test_attention_v(self):
        cls = classify_tensor("layers.0.attn.v_proj.weight", (4096, 4096))
        assert cls.role == TensorRole.ATTENTION_V
    
    def test_attention_o(self):
        cls = classify_tensor("layers.0.attn.o_proj.weight", (4096, 4096))
        assert cls.role == TensorRole.ATTENTION_O
    
    def test_mlp_up(self):
        cls = classify_tensor("layers.0.mlp.up_proj.weight", (4096, 11008))
        assert cls.role == TensorRole.MLP_UP
    
    def test_mlp_down(self):
        cls = classify_tensor("layers.0.mlp.down_proj.weight", (11008, 4096))
        assert cls.role == TensorRole.MLP_DOWN
    
    def test_norm(self):
        cls = classify_tensor("layers.0.norm.weight", (4096,))
        assert cls.role == TensorRole.NORM
    
    def test_output_head(self):
        cls = classify_tensor("lm_head.weight", (4096, 32000))
        assert cls.role == TensorRole.OUTPUT_HEAD
    
    def test_bias(self):
        cls = classify_tensor("layers.0.attn.q_proj.bias", (4096,))
        assert cls.role == TensorRole.BIAS
    
    def test_unknown(self):
        cls = classify_tensor("custom_tensor", (100, 100))
        assert cls.role == TensorRole.UNKNOWN


class TestTensorClassifier:
    def test_classify_model(self):
        tensors = {
            "embed_tokens.weight": np.zeros((32000, 4096)),
            "layers.0.attn.q_proj.weight": np.zeros((4096, 4096)),
            "layers.0.attn.k_proj.weight": np.zeros((4096, 4096)),
            "layers.0.mlp.up_proj.weight": np.zeros((4096, 11008)),
            "layers.0.norm.weight": np.zeros((4096,)),
            "lm_head.weight": np.zeros((4096, 32000)),
        }
        classifier = TensorClassifier(architecture="llama")
        results = classifier.classify_model(tensors)
        
        assert results["embed_tokens.weight"].role == TensorRole.EMBEDDING
        assert results["layers.0.attn.q_proj.weight"].role == TensorRole.ATTENTION_Q
        assert results["layers.0.attn.k_proj.weight"].role == TensorRole.ATTENTION_K
        assert results["layers.0.mlp.up_proj.weight"].role == TensorRole.MLP_UP
        assert results["layers.0.norm.weight"].role == TensorRole.NORM
        assert results["lm_head.weight"].role == TensorRole.OUTPUT_HEAD
    
    def test_role_distribution(self):
        tensors = {
            "a": np.zeros((100, 100)),
            "b": np.zeros((100, 100)),
        }
        # Create with custom roles
        meta = {
            "a": TensorMetadata("a", (100, 100), np.float32, 10000),
            "b": TensorMetadata("b", (100, 100), np.float32, 10000),
        }
        classifier = TensorClassifier()
        
        # Test distribution
        dist = classifier.get_role_distribution({})
        assert dist == {}


# ---------- Storage Tests ----------

class TestMixedPrecisionStorageModel:
    def test_fp16_storage(self):
        model = MixedPrecisionStorageModel()
        assignment = TensorPrecisionAssignment(
            tensor_name="test",
            precision=PrecisionType.FP16,
        )
        cls = TensorClassification(
            tensor_name="test", role=TensorRole.ATTENTION_Q,
            shape=(100, 100), num_parameters=10000, dtype="float32"
        )
        
        comp = model.calculate_tensor_storage(assignment, cls)
        assert comp.precision == PrecisionType.FP16
        assert comp.weight_bytes == 20000  # 10000 * 2
        assert comp.scale_bytes == 0
        assert comp.bits_per_weight == 16.0
    
    def test_int8_storage(self):
        model = MixedPrecisionStorageModel(group_size=128)
        assignment = TensorPrecisionAssignment(
            tensor_name="test",
            precision=PrecisionType.INT8,
            group_size=128,
        )
        cls = TensorClassification(
            tensor_name="test", role=TensorRole.ATTENTION_Q,
            shape=(1000,), num_parameters=1000, dtype="float32"
        )
        
        comp = model.calculate_tensor_storage(assignment, cls)
        assert comp.precision == PrecisionType.INT8
        assert comp.weight_bytes == 1000  # 1 byte per weight
        assert comp.scale_bytes > 0  # Has scale overhead
        assert comp.bits_per_weight > 8.0  # With scale overhead
    
    def test_int4_storage(self):
        model = MixedPrecisionStorageModel(group_size=128)
        assignment = TensorPrecisionAssignment(
            tensor_name="test",
            precision=PrecisionType.INT4,
            group_size=128,
        )
        cls = TensorClassification(
            tensor_name="test", role=TensorRole.ATTENTION_Q,
            shape=(1000,), num_parameters=1000, dtype="float32"
        )
        
        comp = model.calculate_tensor_storage(assignment, cls)
        assert comp.precision == PrecisionType.INT4
        assert comp.weight_bytes == 500  # 2 weights per byte
        assert comp.scale_bytes > 0
        assert comp.bits_per_weight > 4.0
    
    def test_binary_storage(self):
        model = MixedPrecisionStorageModel(group_size=128)
        assignment = TensorPrecisionAssignment(
            tensor_name="test",
            precision=PrecisionType.BINARY,
            group_size=128,
        )
        cls = TensorClassification(
            tensor_name="test", role=TensorRole.ATTENTION_Q,
            shape=(128,), num_parameters=128, dtype="float32"
        )
        
        comp = model.calculate_tensor_storage(assignment, cls)
        assert comp.precision == PrecisionType.BINARY
        assert comp.weight_bytes == 16  # 128 bits = 16 bytes
        assert comp.scale_bytes == 2  # 1 FP16 scale
        assert abs(comp.bits_per_weight - 1.125) < 0.01  # 1.125 bpw for binary


class TestStorageAccounting:
    def test_plan_storage_calculation(self):
        model = MixedPrecisionStorageModel(group_size=128)
        
        # Create plan
        plan = create_uniform_plan("test", ["t1", "t2"], PrecisionType.INT8, group_size=128)
        
        classifications = {
            "t1": TensorClassification("t1", TensorRole.ATTENTION_Q, (1000,), 1000, "float32"),
            "t2": TensorClassification("t2", TensorRole.MLP_UP, (1000,), 1000, "float32"),
        }
        
        storage = model.calculate_plan_storage(plan, classifications)
        
        assert storage.total_tensors == 2
        assert storage.total_parameters == 2000
        assert storage.total_bytes > 0
        assert storage.average_bits_per_weight > 8.0
        assert storage.compression_ratio > 1.0
        assert storage.precision_distribution["int8"] == 2


# ---------- Error Analysis Tests ----------

class TestErrorMetrics:
    def test_perfect_reconstruction(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mae, mse, rmse, max_err, rel_err = calculate_error_metrics(a, a)
        assert mae == 0.0
        assert mse == 0.0
        assert rmse == 0.0
        assert max_err == 0.0
        assert rel_err == 0.0
    
    def test_known_error(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        mae, mse, rmse, max_err, rel_err = calculate_error_metrics(a, b)
        assert mae == 1.0
        assert mse == 1.0
        assert rmse == 1.0
        assert max_err == 1.0
    
    def test_relative_error(self):
        a = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
        b = np.array([0.0, 0.0], dtype=np.float32)
        mae, mse, rmse, max_err, rel_err = calculate_error_metrics(a, b)
        assert abs(rel_err - 1.0) < 1e-6


# ---------- Search Tests ----------

class TestSearchConstraint:
    def test_default_constraints(self):
        constraint = SearchConstraint()
        assert constraint.max_storage_bytes is None
        assert constraint.max_average_bits_per_weight is None
    
    def test_custom_constraints(self):
        constraint = SearchConstraint(
            max_storage_gb=4.0,
            max_average_bits_per_weight=6.0,
            max_mae=0.01,
        )
        assert constraint.max_storage_gb == 4.0
        assert constraint.max_average_bits_per_weight == 6.0
        assert constraint.max_mae == 0.01


# ---------- Scenario Tests ----------

class TestModelScaleScenario:
    def test_default_scenarios(self):
        scenarios = get_default_scenarios()
        assert len(scenarios) == 4
        
        names = [s.name for s in scenarios]
        assert "7B" in names
        assert "13B" in names
        assert "27B" in names
        assert "70B" in names
    
    def test_7b_scenario(self):
        scenario = next(s for s in get_default_scenarios() if s.name == "7B")
        assert scenario.total_parameters == 6_700_000_000
        assert scenario.layer_count == 32
        assert scenario.hidden_size == 4096
    
    def test_parameter_distribution(self):
        scenario = next(s for s in get_default_scenarios() if s.name == "7B")
        dist = estimate_parameter_distribution(scenario)
        
        # Check all roles have positive params
        for role, params in dist.items():
            assert params > 0
        
        # Total should roughly match (within 25% - these are rough estimates)
        total = sum(dist.values())
        assert abs(total - scenario.total_parameters) < scenario.total_parameters * 0.25


class TestClassificationsFromScenario:
    def test_create_classifications(self):
        scenario = next(s for s in get_default_scenarios() if s.name == "7B")
        classifications = create_classifications_from_scenario(scenario)
        
        assert len(classifications) > 0
        for name, cls in classifications.items():
            assert cls.num_parameters > 0
            assert cls.role != TensorRole.UNKNOWN
            assert cls.shape is not None
    
    def test_uniform_plan_creation(self):
        scenario = next(s for s in get_default_scenarios() if s.name == "7B")
        classifications = create_classifications_from_scenario(scenario)
        
        plan = create_uniform_fp16_plan(classifications)
        assert plan.total_tensors() == len(classifications)
        assert all(a.precision == PrecisionType.FP16 for a in plan.assignments.values())


class TestMixedStrategies:
    def test_create_strategies(self):
        scenario = next(s for s in get_default_scenarios() if s.name == "7B")
        classifications = create_classifications_from_scenario(scenario)
        
        strategies = create_mixed_precision_strategies(classifications)
        
        assert "conservative" in strategies
        assert "aggressive" in strategies
        assert "int4_heavy" in strategies
        assert "binary_mlp" in strategies
        
        # Check conservative: norms are FP16
        for name, cls in classifications.items():
            if cls.role == TensorRole.NORM:
                assignment = strategies["conservative"].get_assignment(name)
                assert assignment.precision == PrecisionType.FP16
        
        # Check aggressive: attention is INT4
        for name, cls in classifications.items():
            if cls.role in (TensorRole.ATTENTION_Q, TensorRole.ATTENTION_K,
                           TensorRole.ATTENTION_V, TensorRole.ATTENTION_O):
                assignment = strategies["aggressive"].get_assignment(name)
                assert assignment.precision == PrecisionType.INT4


# ---------- Edge Cases ----------

class TestEdgeCases:
    def test_empty_plan(self):
        plan = MixedPrecisionPlan(model_identifier="empty")
        assert plan.total_tensors() == 0
        assert plan.get_precision("nonexistent") is None
    
    def test_invalid_precision(self):
        with pytest.raises(ValueError):
            PrecisionType("invalid")
    
    def test_storage_impossible_constraints(self):
        constraint = SearchConstraint(max_storage_bytes=0)
        # Should be handled gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])