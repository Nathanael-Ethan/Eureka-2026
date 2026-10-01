import pytest
import numpy as np
from src.ldmark.analysis.parameters import (
    count_parameters,
    count_parameters_from_dict,
    count_parameters_from_numpy,
    get_parameter_summary,
    estimate_parameters_from_config,
    compare_parameter_counts,
)
from src.ldmark.analysis.tensor import create_tensor_info, analyze_numpy_dict
from src.ldmark.analysis.models import DType, TensorTrainableStatus, ParameterCounts


class TestCountParameters:
    def test_basic_count(self):
        tensors = [
            create_tensor_info("layer1.weight", [256, 512], DType.FP16, TensorTrainableStatus.TRAINABLE),
            create_tensor_info("layer2.weight", [512, 256], DType.FP16, TensorTrainableStatus.FROZEN),
            create_tensor_info("bias", [256], DType.FP32, TensorTrainableStatus.UNKNOWN),
        ]
        counts = count_parameters(tensors)
        assert counts.total == 256 * 512 + 512 * 256 + 256
        assert counts.by_dtype[DType.FP16] == 256 * 512 + 512 * 256
        assert counts.by_dtype[DType.FP32] == 256
        assert counts.trainable == 256 * 512
        assert counts.non_trainable == 512 * 256
        assert counts.unknown_trainable == 256

    def test_empty_list(self):
        counts = count_parameters([])
        assert counts.total == 0


class TestCountParametersFromNumpy:
    def test_from_numpy_dict(self):
        tensors = {
            "layer1.weight": np.random.randn(256, 512).astype(np.float16),
            "layer2.weight": np.random.randn(512, 256).astype(np.float32),
        }
        counts = count_parameters_from_numpy(tensors)
        assert counts.total == 256 * 512 + 512 * 256
        assert counts.by_dtype[DType.FP16] == 256 * 512
        assert counts.by_dtype[DType.FP32] == 512 * 256


class TestGetParameterSummary:
    def test_summary(self):
        counts = ParameterCounts()
        counts.add_tensor("layer1.weight", 1000, DType.FP16, TensorTrainableStatus.TRAINABLE)
        counts.add_tensor("layer2.weight", 2000, DType.FP16, TensorTrainableStatus.FROZEN)
        counts.add_tensor("layer3.bias", 500, DType.FP32, TensorTrainableStatus.UNKNOWN)

        summary = get_parameter_summary(counts)
        assert summary["total_parameters"] == 3500
        assert summary["trainable_parameters"] == 1000
        assert summary["frozen_parameters"] == 2000
        assert summary["unknown_trainable_parameters"] == 500
        assert summary["by_dtype"]["float16"] == 3000
        assert summary["by_dtype"]["float32"] == 500
        assert summary["num_tensors"] == 3
        assert len(summary["largest_tensors"]) == 3


class TestEstimateParametersFromConfig:
    def test_llama_config(self):
        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
        }
        estimated = estimate_parameters_from_config(config)
        assert estimated is not None
        assert estimated > 0

    def test_missing_config(self):
        config = {"model_type": "unknown"}
        estimated = estimate_parameters_from_config(config)
        assert estimated is None

    def test_minimal_config(self):
        config = {
            "hidden_size": 256,
            "num_hidden_layers": 4,
        }
        estimated = estimate_parameters_from_config(config)
        assert estimated is not None


class TestCompareParameterCounts:
    def test_match(self):
        actual = ParameterCounts()
        actual.add_tensor("w", 1000, DType.FP16, TensorTrainableStatus.TRAINABLE)
        result = compare_parameter_counts(actual, 1000)
        assert result["match"] is True
        assert result["percent_difference"] == 0.0

    def test_mismatch(self):
        actual = ParameterCounts()
        actual.add_tensor("w", 1000, DType.FP16, TensorTrainableStatus.TRAINABLE)
        result = compare_parameter_counts(actual, 2000)
        assert result["match"] is False
        assert result["percent_difference"] == -50.0