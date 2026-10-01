import pytest
import json
import numpy as np
from pathlib import Path
from src.ldmark.model_io import ModelIO, inspect_model, load_and_analyze
from src.ldmark.analysis import ModelAnalysis
from src.ldmark.file_io import ModelFormat


class TestModelIO:
    def test_inspect_numpy_metadata_only(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        analysis = io.inspect(p, metadata_only=True)

        assert isinstance(analysis, ModelAnalysis)
        assert analysis.parameter_counts.total == 256 * 512
        assert analysis.model_id == "model"

    def test_inspect_numpy_full(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        analysis = io.inspect(p, metadata_only=False)

        assert isinstance(analysis, ModelAnalysis)
        assert analysis.parameter_counts.total == 256 * 512

    def test_inspect_config(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))

        io = ModelIO()
        analysis = io.inspect(p, metadata_only=True)

        assert isinstance(analysis, ModelAnalysis)
        assert analysis.architecture.model_architecture == "llama"
        assert analysis.architecture.num_layers == 32
        assert analysis.architecture.hidden_size == 4096

    def test_load_and_analyze_numpy(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        analysis = io.load_and_analyze(p)

        assert isinstance(analysis, ModelAnalysis)
        assert analysis.parameter_counts.total == 256 * 512
        assert len(analysis.storage_estimates) > 0
        assert len(analysis.compression_estimates) > 0

    def test_get_metadata(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        metadata = io.get_metadata(p)

        assert metadata.total_parameters == 256 * 512
        assert metadata.tensor_count == 1

    def test_get_loaded_model(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        loaded = io.get_loaded_model(p)

        assert loaded.total_parameters == 256 * 512
        assert loaded.tensor_count == 1

    def test_inspect_model_function(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        analysis = inspect_model(p, metadata_only=True)
        assert isinstance(analysis, ModelAnalysis)
        assert analysis.parameter_counts.total == 256 * 512

    def test_load_and_analyze_function(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        analysis = load_and_analyze(p)
        assert isinstance(analysis, ModelAnalysis)
        assert analysis.parameter_counts.total == 256 * 512


class TestModelIOWithConfig:
    def test_numpy_with_config(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 1000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arr = np.random.randn(256, 512).astype(np.float16)
        np.save(tmp_path / "weights.npy", arr)

        io = ModelIO()
        analysis = io.inspect(tmp_path, metadata_only=True)

        assert analysis.architecture.model_architecture == "llama"
        assert analysis.architecture.num_layers == 2
        assert analysis.architecture.hidden_size == 128
        assert analysis.parameter_counts.total == 256 * 512

    def test_analysis_has_storage_estimates(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        analysis = io.inspect(p, metadata_only=True)

        assert len(analysis.storage_estimates) > 0
        fp16_est = next((e for e in analysis.storage_estimates if e.dtype.value == "float16"), None)
        assert fp16_est is not None

    def test_analysis_has_compression_estimates(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        analysis = io.inspect(p, metadata_only=True)

        assert len(analysis.compression_estimates) > 0

    def test_analysis_json_serialization(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        io = ModelIO()
        analysis = io.inspect(p, metadata_only=True)

        json_str = analysis.to_json()
        assert "parameter_counts" in json_str
        assert "storage_estimates" in json_str