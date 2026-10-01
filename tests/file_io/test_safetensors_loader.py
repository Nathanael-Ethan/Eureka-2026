import pytest
import json
import numpy as np
from pathlib import Path
from src.ldmark.file_io.safetensors_loader import SafetensorsLoader
from src.ldmark.file_io.formats import ModelFormat, ModelSource
from src.ldmark.file_io.metadata import TensorDtype
from src.ldmark.file_io.loader import CorruptFileError
from src.ldmark.file_io.capabilities import is_safetensors_available


safetensors = pytest.importorskip("safetensors", reason="safetensors not available")


class TestSafetensorsLoader:
    def test_can_load_safetensors(self, tmp_path):
        loader = SafetensorsLoader()
        source = ModelSource(tmp_path / "model.safetensors", ModelFormat.SAFETENSORS)
        assert loader.can_load(source) is True

    def test_cannot_load_numpy(self, tmp_path):
        loader = SafetensorsLoader()
        source = ModelSource(tmp_path / "model.npy", ModelFormat.NUMPY)
        assert loader.can_load(source) is False

    def test_load_safetensors_file(self, tmp_path):
        from safetensors.numpy import save_file

        tensors = {
            "weight1": np.random.randn(256, 512).astype(np.float16),
            "weight2": np.random.randn(512, 256).astype(np.float32),
        }
        p = tmp_path / "model.safetensors"
        save_file(tensors, str(p))

        loader = SafetensorsLoader()
        source = ModelSource(p, ModelFormat.SAFETENSORS)
        loaded = loader.load(source)

        assert loaded.total_parameters == 256 * 512 + 512 * 256
        assert loaded.tensor_count == 2

    def test_load_metadata_only(self, tmp_path):
        from safetensors.numpy import save_file

        tensors = {
            "weight1": np.random.randn(256, 512).astype(np.float16),
        }
        p = tmp_path / "model.safetensors"
        save_file(tensors, str(p))

        loader = SafetensorsLoader(metadata_only=True)
        source = ModelSource(p, ModelFormat.SAFETENSORS)
        metadata = loader.load_metadata(source)

        assert metadata.total_parameters == 256 * 512
        assert metadata.tensor_count == 1
        assert metadata.tensors[0].dtype == TensorDtype.FLOAT16

    def test_dtype_preservation(self, tmp_path):
        from safetensors.numpy import save_file

        tensors = {
            "weight": np.random.randn(256, 512).astype(np.float16),
        }
        p = tmp_path / "model.safetensors"
        save_file(tensors, str(p))

        loader = SafetensorsLoader()
        source = ModelSource(p, ModelFormat.SAFETENSORS)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("weight")
        assert tensor.metadata.dtype == TensorDtype.FLOAT16

    def test_config_from_directory(self, tmp_path):
        from safetensors.numpy import save_file

        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        tensors = {"weight": np.random.randn(256, 512).astype(np.float16)}
        save_file(tensors, str(tmp_path / "model.safetensors"))

        loader = SafetensorsLoader()
        source = ModelSource(tmp_path, ModelFormat.SAFETENSORS, is_directory=True)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "LlamaForCausalLM"
        assert loaded.metadata.num_layers == 32

    def test_nonexistent_file_raises(self, tmp_path):
        loader = SafetensorsLoader()
        source = ModelSource(tmp_path / "nonexistent.safetensors", ModelFormat.SAFETENSORS)
        with pytest.raises(FileNotFoundError):
            loader.load(source)