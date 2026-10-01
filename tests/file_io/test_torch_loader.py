import pytest
import json
import numpy as np
from pathlib import Path
from src.ldmark.file_io.torch_loader import TorchLoader
from src.ldmark.file_io.formats import ModelFormat, ModelSource
from src.ldmark.file_io.metadata import TensorDtype
from src.ldmark.file_io.loader import MissingDependencyError, CorruptFileError
from src.ldmark.file_io.capabilities import is_torch_available


torch = pytest.importorskip("torch", reason="PyTorch not available")


class TestTorchLoader:
    def test_can_load_pytorch(self, tmp_path):
        loader = TorchLoader()
        source = ModelSource(tmp_path / "model.bin", ModelFormat.PYTORCH)
        assert loader.can_load(source) is True

    def test_cannot_load_numpy(self, tmp_path):
        loader = TorchLoader()
        source = ModelSource(tmp_path / "model.npy", ModelFormat.NUMPY)
        assert loader.can_load(source) is False

    def test_load_state_dict(self, tmp_path):
        state_dict = {
            "layer1.weight": torch.randn(256, 512).to(torch.float16),
            "layer2.weight": torch.randn(512, 256).to(torch.float32),
        }
        p = tmp_path / "model.bin"
        torch.save(state_dict, p)

        loader = TorchLoader()
        source = ModelSource(p, ModelFormat.PYTORCH)
        loaded = loader.load(source)

        assert loaded.total_parameters == 256 * 512 + 512 * 256
        assert loaded.tensor_count == 2

    def test_load_nested_state_dict(self, tmp_path):
        state_dict = {
            "state_dict": {
                "layer1.weight": torch.randn(256, 512).to(torch.float16),
            }
        }
        p = tmp_path / "model.bin"
        torch.save(state_dict, p)

        loader = TorchLoader()
        source = ModelSource(p, ModelFormat.PYTORCH)
        loaded = loader.load(source)

        assert loaded.total_parameters == 256 * 512

    def test_load_metadata_only(self, tmp_path):
        state_dict = {
            "layer1.weight": torch.randn(256, 512).to(torch.float16),
        }
        p = tmp_path / "model.bin"
        torch.save(state_dict, p)

        loader = TorchLoader(metadata_only=True)
        source = ModelSource(p, ModelFormat.PYTORCH)
        metadata = loader.load_metadata(source)

        assert metadata.total_parameters == 256 * 512
        assert metadata.tensor_count == 1

    def test_dtype_preservation(self, tmp_path):
        state_dict = {
            "weight": torch.randn(256, 512).to(torch.float16),
        }
        p = tmp_path / "model.bin"
        torch.save(state_dict, p)

        loader = TorchLoader()
        source = ModelSource(p, ModelFormat.PYTORCH)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("weight")
        assert tensor.metadata.dtype == TensorDtype.FLOAT16

    def test_config_from_directory(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        state_dict = {"weight": torch.randn(256, 512).to(torch.float16)}
        torch.save(state_dict, tmp_path / "pytorch_model.bin")

        loader = TorchLoader()
        source = ModelSource(tmp_path, ModelFormat.PYTORCH, is_directory=True)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "LlamaForCausalLM"
        assert loaded.metadata.num_layers == 32

    def test_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "model.bin"
        p.write_bytes(b"not a valid torch file")

        loader = TorchLoader()
        source = ModelSource(p, ModelFormat.PYTORCH)
        with pytest.raises(CorruptFileError):
            loader.load(source)

    def test_nonexistent_file_raises(self, tmp_path):
        loader = TorchLoader()
        source = ModelSource(tmp_path / "nonexistent.bin", ModelFormat.PYTORCH)
        with pytest.raises(FileNotFoundError):
            loader.load(source)