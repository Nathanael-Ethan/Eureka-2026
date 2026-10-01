import pytest
import numpy as np
import json
from pathlib import Path
from src.ldmark.file_io.numpy_loader import NumpyLoader
from src.ldmark.file_io.formats import ModelFormat, ModelSource
from src.ldmark.file_io.metadata import TensorDtype
from src.ldmark.file_io.loader import CorruptFileError, MalformedStateDictError


class TestNumpyLoader:
    def test_can_load_npy(self, tmp_path):
        loader = NumpyLoader()
        source = ModelSource(tmp_path / "model.npy", ModelFormat.NUMPY)
        assert loader.can_load(source) is True

    def test_cannot_load_pytorch(self, tmp_path):
        loader = NumpyLoader()
        source = ModelSource(tmp_path / "model.bin", ModelFormat.PYTORCH)
        assert loader.can_load(source) is False

    def test_load_npy_file(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        assert loaded.model_id == "model"
        assert loaded.source_format == ModelFormat.NUMPY
        assert loaded.total_parameters == 256 * 512
        assert loaded.tensor_count == 1
        assert "model" in loaded.tensor_names

    def test_load_npz_file(self, tmp_path):
        arr1 = np.random.randn(256, 512).astype(np.float16)
        arr2 = np.random.randn(512, 256).astype(np.float32)
        p = tmp_path / "model.npz"
        np.savez(p, weight1=arr1, weight2=arr2)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        assert loaded.total_parameters == 256 * 512 + 512 * 256
        assert loaded.tensor_count == 2

    def test_load_npy_directory(self, tmp_path):
        arr1 = np.random.randn(256, 512).astype(np.float16)
        arr2 = np.random.randn(512, 256).astype(np.float32)
        np.save(tmp_path / "layer1.npy", arr1)
        np.save(tmp_path / "layer2.npy", arr2)

        loader = NumpyLoader()
        source = ModelSource(tmp_path, ModelFormat.NUMPY, is_directory=True)
        loaded = loader.load(source)

        assert loaded.total_parameters == 256 * 512 + 512 * 256
        assert loaded.tensor_count == 2

    def test_load_metadata_only(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader(metadata_only=True)
        source = ModelSource(p, ModelFormat.NUMPY)
        metadata = loader.load_metadata(source)

        assert metadata.total_parameters == 256 * 512
        assert metadata.tensor_count == 1
        assert len(metadata.tensors) == 1
        assert metadata.tensors[0].name == "model"
        assert metadata.tensors[0].dtype == TensorDtype.FLOAT16

    def test_dtype_preservation_float16(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("model")
        assert tensor.metadata.dtype == TensorDtype.FLOAT16

    def test_dtype_preservation_float32(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float32)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("model")
        assert tensor.metadata.dtype == TensorDtype.FLOAT32

    def test_shape_preservation(self, tmp_path):
        arr = np.random.randn(128, 256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        tensor = loaded.get_tensor("model")
        assert tensor.metadata.shape == [128, 256, 512]

    def test_config_from_directory(self, tmp_path):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        arr = np.random.randn(256, 512).astype(np.float16)
        np.save(tmp_path / "weights.npy", arr)

        loader = NumpyLoader()
        source = ModelSource(tmp_path, ModelFormat.NUMPY, is_directory=True)
        loaded = loader.load(source)

        assert loaded.metadata.architecture == "LlamaForCausalLM"
        assert loaded.metadata.num_layers == 32
        assert loaded.metadata.hidden_size == 4096
        assert loaded.metadata.num_kv_heads == 8

    def test_corrupt_file_raises(self, tmp_path):
        p = tmp_path / "model.npy"
        p.write_bytes(b"not a valid npy file")

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        with pytest.raises(CorruptFileError):
            loader.load(source)

    def test_nonexistent_file_raises(self, tmp_path):
        loader = NumpyLoader()
        source = ModelSource(tmp_path / "nonexistent.npy", ModelFormat.NUMPY)
        with pytest.raises(FileNotFoundError):
            loader.load(source)

    def test_empty_directory_raises(self, tmp_path):
        loader = NumpyLoader()
        source = ModelSource(tmp_path, ModelFormat.NUMPY, is_directory=True)
        with pytest.raises(MalformedStateDictError):
            loader.load(source)

    def test_to_state_dict(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        state_dict = loaded.to_state_dict()
        assert "model" in state_dict
        assert state_dict["model"].shape == (256, 512)

    def test_get_summary(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loader = NumpyLoader()
        source = ModelSource(p, ModelFormat.NUMPY)
        loaded = loader.load(source)

        summary = loaded.get_summary()
        assert summary["model_id"] == "model"
        assert summary["total_parameters"] == 256 * 512
        assert summary["tensor_count"] == 1