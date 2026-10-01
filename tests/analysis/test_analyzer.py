import pytest
import numpy as np
from src.ldmark.analysis.analyzer import ModelAnalyzer, analyze_model, analyze_state_dict
from src.ldmark.analysis.io import create_synthetic_model, create_synthetic_state_dict
from src.ldmark.analysis.models import DType, ModelAnalysis


class TestModelAnalyzer:
    def test_analyze_numpy_tensors(self):
        model_data = create_synthetic_model(
            num_layers=2,
            hidden_size=128,
            intermediate_size=512,
            num_heads=4,
            vocab_size=1000,
        )
        analyzer = ModelAnalyzer("test-model", "/fake/path")
        analysis = analyzer.analyze_numpy_tensors(model_data["tensors"], config=model_data["config"])

        assert isinstance(analysis, ModelAnalysis)
        assert analysis.model_id == "test-model"
        assert analysis.parameter_counts.total > 0
        assert analysis.architecture.model_architecture == "llama"
        assert analysis.architecture.num_layers == 2
        assert analysis.architecture.hidden_size == 128
        assert len(analysis.storage_estimates) > 0
        assert len(analysis.compression_estimates) > 0
        assert analysis.runtime_memory is not None

    def test_analyze_state_dict(self):
        state_dict = create_synthetic_state_dict(param_count=1_000_000, num_tensors=10)
        analyzer = ModelAnalyzer("test-model", "/fake/path")
        analysis = analyzer.analyze_from_state_dict(state_dict)

        assert analysis.parameter_counts.total == 1_000_000
        assert len(analysis.tensors) == 10

    def test_analyze_tensors_with_trainable(self):
        state_dict = create_synthetic_state_dict(param_count=1_000_000, num_tensors=5)
        trainable_names = {"layer.0.weight", "layer.1.weight"}
        analyzer = ModelAnalyzer("test-model", "/fake/path")
        analysis = analyzer.analyze_tensors(state_dict, trainable_names=trainable_names)

        assert analysis.parameter_counts.trainable > 0
        assert analysis.parameter_counts.non_trainable > 0

    def test_get_summary(self):
        model_data = create_synthetic_model(num_layers=2, hidden_size=128, num_heads=4)
        analyzer = ModelAnalyzer("test-model", "/fake/path")
        analyzer.analyze_numpy_tensors(model_data["tensors"], config=model_data["config"])
        summary = analyzer.get_summary()

        assert summary["model_id"] == "test-model"
        assert summary["total_parameters"] > 0
        assert summary["architecture"] == "llama"
        assert summary["num_layers"] == 2
        assert summary["hidden_size"] == 128
        assert summary["weight_dtype"] == "float16"
        assert summary["estimated_fp16_size_gb"] > 0
        assert summary["estimated_int4_size_gb"] > 0


class TestAnalyzeModel:
    def test_analyze_model_function(self):
        model_data = create_synthetic_model(num_layers=1, hidden_size=64, num_heads=2)
        analysis = analyze_model(
            model_data["tensors"],
            model_id="func-test",
            source_path="/fake/path",
            config=model_data["config"],
        )
        assert analysis.model_id == "func-test"
        assert analysis.parameter_counts.total > 0


class TestAnalyzeStateDict:
    def test_analyze_state_dict_function(self):
        state_dict = create_synthetic_state_dict(param_count=500_000, num_tensors=5)
        analysis = analyze_state_dict(state_dict, model_id="state-dict-test")
        assert analysis.model_id == "state-dict-test"
        assert analysis.parameter_counts.total == 500_000


class TestFileSizeComparison:
    def test_warnings_on_file_size(self):
        model_data = create_synthetic_model(num_layers=1, hidden_size=64, num_heads=2)
        analyzer = ModelAnalyzer("test", "/fake/path")
        analyzer.actual_file_size_bytes = 100_000_000
        analysis = analyzer.analyze_numpy_tensors(model_data["tensors"], config=model_data["config"])

        file_size_warnings = [w for w in analysis.warnings if "File size vs theoretical" in w]
        assert len(file_size_warnings) == 1