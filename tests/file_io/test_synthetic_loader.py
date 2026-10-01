import pytest
import json
import numpy as np
from pathlib import Path
from src.ldmark.file_io.synthetic_loader import SyntheticLoader
from src.ldmark.file_io.formats import ModelFormat, ModelSource
from src.ldmark.file_io.metadata import TensorDtype
from src.ldmark.file_io.loader import CorruptFileError, MalformedStateDictError


class TestSyntheticLoader:
    def test_can_load_synthetic(self, tmp_path):
        loader = SyntheticLoader()
        source = ModelSource(tmp_path / "synthetic.json", ModelFormat.SYNTHETIC)
        assert loader.can_load(source) is True

    def test_cannot_load_numpy(self, tmp_path):
        loader = SyntheticLoader()
        source = ModelSource(tmp_path / "model.npy", ModelFormat.NUMPY)
        assert loader.can_load(source) is False

    def test_load_synthetic_file(self, tmp_path):
        spec = {
            "model_id": "test-synthetic",
            "config": {
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 2,
                "hidden_size": 128,
                "num_attention_heads": 4,
            },
            "tensors": [
                {"name": "layer1.weight", "shape": [128, 256], "dtype": "float16"},
                {"name": "layer2.weight", "shape": [256, 128], "dtype": "float32"},
            ],
        }
        p = tmp_path / "synthetic.json"
        p.write_text(json.dumps(spec))

        loader = SyntheticLoader()
        source = ModelSource(p, ModelFormat.SYNTHETIC)
        loaded = loader.load(source)

        assert loaded.model_id == "test-synthetic"
        assert loaded.metadata.architecture == "LlamaForCausalLM"
        assert loaded.metadata.num_layers == 2
        assert loaded.total_parameters == 128 * 256 + 256 * 128
        assert loaded.tensor_count == 2

    def test_load_metadata_only(self, tmp_path):
        spec = {
            "model_id": "test-synthetic",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "tensors": [
                {"name": "weight", "shape": [256, 512], "dtype": "float16"},
            ],
        }
        p = tmp_path / "synthetic.json"
        p.write_text(json.dumps(spec))

        loader = SyntheticLoader(metadata_only=True)
        source = ModelSource(p, ModelFormat.SYNTHETIC)
        metadata = loader.load_metadata(source)

        assert metadata.total_parameters == 256 * 512
        assert metadata.tensor_count == 1
        assert metadata.tensors[0].dtype == TensorDtype.FLOAT16

    def test_dtype_preservation(self, tmp_path):
        spec = {
            "model_id": "test",
            "config": {},
            "tensors": [
                {"name": "weight", "shape": [256, 512], "dtype": "float32"},
            ],
        }
        p = tmp_path / "synthetic.json"
        p.write_text(json.dumps(spec))

        loader = SyntheticLoader()
        source = ModelSource(p, ModelFormat.SYNTHETIC)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("weight")
        assert tensor.metadata.dtype == TensorDtype.FLOAT32

    def test_shape_preservation(self, tmp_path):
        spec = {
            "model_id": "test",
            "config": {},
            "tensors": [
                {"name": "weight", "shape": [128, 256, 512], "dtype": "float16"},
            ],
        }
        p = tmp_path / "synthetic.json"
        p.write_text(json.dumps(spec))

        loader = SyntheticLoader()
        source = ModelSource(p, ModelFormat.SYNTHETIC)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("weight")
        assert tensor.metadata.shape == [128, 256, 512]

    def test_load_from_directory(self, tmp_path):
        spec = {
            "model_id": "test-synthetic",
            "config": {"architectures": ["LlamaForCausalLM"]},
            "tensors": [
                {"name": "weight", "shape": [256, 512], "dtype": "float16"},
            ],
        }
        (tmp_path / "synthetic_spec.json").write_text(json.dumps(spec))

        loader = SyntheticLoader()
        source = ModelSource(tmp_path, ModelFormat.SYNTHETIC, is_directory=True)
        loaded = loader.load(source)

        assert loaded.model_id == "test-synthetic"
        assert loaded.total_parameters == 256 * 512

    def test_corrupt_json_raises(self, tmp_path):
        p = tmp_path / "synthetic.json"
        p.write_text("not valid json")

        loader = SyntheticLoader()
        source = ModelSource(p, ModelFormat.SYNTHETIC)
        with pytest.raises(CorruptFileError):
            loader.load(source)

    def test_nonexistent_file_raises(self, tmp_path):
        loader = SyntheticLoader()
        source = ModelSource(tmp_path / "nonexistent.json", ModelFormat.SYNTHETIC)
        with pytest.raises(FileNotFoundError):
            loader.load(source)

    def test_missing_spec_in_directory(self, tmp_path):
        loader = SyntheticLoader()
        source = ModelSource(tmp_path, ModelFormat.SYNTHETIC, is_directory=True)
        with pytest.raises(MalformedStateDictError):
            loader.load(source)