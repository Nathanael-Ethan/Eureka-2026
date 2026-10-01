import pytest
import json
from pathlib import Path
from src.ldmark.file_io.config_loader import ConfigLoader
from src.ldmark.file_io.formats import ModelFormat, ModelSource
from src.ldmark.file_io.loader import CorruptFileError, MalformedStateDictError


class TestConfigLoader:
    def test_can_load_json(self, tmp_path):
        loader = ConfigLoader()
        source = ModelSource(tmp_path / "config.json", ModelFormat.JSON_CONFIG)
        assert loader.can_load(source) is True

    def test_cannot_load_numpy(self, tmp_path):
        loader = ConfigLoader()
        source = ModelSource(tmp_path / "model.npy", ModelFormat.NUMPY)
        assert loader.can_load(source) is False

    def test_load_config_file(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))

        loader = ConfigLoader()
        source = ModelSource(p, ModelFormat.JSON_CONFIG)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "LlamaForCausalLM"
        assert loaded.metadata.num_layers == 32
        assert loaded.metadata.hidden_size == 4096
        assert loaded.metadata.num_kv_heads == 8
        assert loaded.metadata.vocab_size == 32000
        assert loaded.metadata.attention_type == "gqa"

    def test_load_config_from_directory(self, tmp_path):
        config = {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        loader = ConfigLoader()
        source = ModelSource(tmp_path, ModelFormat.JSON_CONFIG, is_directory=True)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "MistralForCausalLM"
        assert loaded.metadata.num_layers == 32

    def test_load_metadata_only(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))

        loader = ConfigLoader(metadata_only=True)
        source = ModelSource(p, ModelFormat.JSON_CONFIG)
        metadata = loader.load_metadata(source)

        assert metadata.architecture == "LlamaForCausalLM"
        assert metadata.num_layers == 32
        assert metadata.total_parameters == 0
        assert metadata.tensor_count == 0

    def test_unknown_architecture(self, tmp_path):
        config = {"model_type": "custom_model"}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))

        loader = ConfigLoader()
        source = ModelSource(p, ModelFormat.JSON_CONFIG)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "custom_model"

    def test_empty_config(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({}))

        loader = ConfigLoader()
        source = ModelSource(p, ModelFormat.JSON_CONFIG)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "unknown"

    def test_corrupt_json_raises(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("not valid json")

        loader = ConfigLoader()
        source = ModelSource(p, ModelFormat.JSON_CONFIG)
        with pytest.raises(CorruptFileError):
            loader.load(source)

    def test_nonexistent_file_raises(self, tmp_path):
        loader = ConfigLoader()
        source = ModelSource(tmp_path / "nonexistent.json", ModelFormat.JSON_CONFIG)
        with pytest.raises(FileNotFoundError):
            loader.load(source)

    def test_missing_config_in_directory(self, tmp_path):
        loader = ConfigLoader()
        source = ModelSource(tmp_path, ModelFormat.JSON_CONFIG, is_directory=True)
        with pytest.raises(MalformedStateDictError):
            loader.load(source)