import pytest
import json
import numpy as np
from pathlib import Path
from src.ldmark.file_io.factory import (
    LoaderFactory,
    get_default_factory,
    load_model,
    load_model_metadata,
)
from src.ldmark.file_io.formats import ModelFormat
from src.ldmark.file_io.loader import UnsupportedFormatError, MissingDependencyError
from src.ldmark.file_io.capabilities import is_torch_available, is_safetensors_available


class TestLoaderFactory:
    def test_factory_creation(self):
        factory = LoaderFactory()
        assert factory is not None

    def test_get_available_formats(self):
        factory = LoaderFactory()
        formats = factory.get_available_formats()
        assert "numpy" in formats
        assert "json_config" in formats

    def test_get_loader_numpy(self):
        factory = LoaderFactory()
        loader = factory.get_loader(ModelFormat.NUMPY)
        assert loader is not None

    def test_get_loader_json_config(self):
        factory = LoaderFactory()
        loader = factory.get_loader(ModelFormat.JSON_CONFIG)
        assert loader is not None

    def test_get_loader_unsupported_format(self):
        factory = LoaderFactory()
        with pytest.raises(UnsupportedFormatError):
            factory.get_loader(ModelFormat.UNKNOWN)

    def test_get_loader_missing_torch(self):
        factory = LoaderFactory()
        if not is_torch_available():
            with pytest.raises(MissingDependencyError):
                factory.get_loader(ModelFormat.PYTORCH)

    def test_get_loader_missing_safetensors(self):
        factory = LoaderFactory()
        if not is_safetensors_available():
            with pytest.raises(MissingDependencyError):
                factory.get_loader(ModelFormat.SAFETENSORS)

    def test_load_numpy_model(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        factory = LoaderFactory()
        loaded = factory.load(p)
        assert loaded.total_parameters == 256 * 512

    def test_load_numpy_metadata_only(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        factory = LoaderFactory()
        loaded = factory.load(p, metadata_only=True)
        assert loaded.total_parameters == 256 * 512
        assert loaded.tensor_count == 0

    def test_load_config_model(self, tmp_path):
        config = {"model_type": "llama", "num_hidden_layers": 32}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))

        factory = LoaderFactory()
        loaded = factory.load(p)
        assert loaded.metadata.architecture == "llama"

    def test_load_metadata(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        factory = LoaderFactory()
        metadata = factory.load_metadata(p)
        assert metadata.total_parameters == 256 * 512

    def test_can_load_numpy(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        factory = LoaderFactory()
        assert factory.can_load(p) is True

    def test_can_load_nonexistent(self):
        factory = LoaderFactory()
        assert factory.can_load("/nonexistent/path") is False

    def test_register_custom_loader(self, tmp_path):
        from src.ldmark.file_io.loader import BaseModelLoader

        class CustomLoader(BaseModelLoader):
            def can_load(self, source):
                return True

            def load(self, source):
                pass

            def load_metadata(self, source):
                pass

        factory = LoaderFactory()
        custom_loader = CustomLoader()
        factory.register_loader(ModelFormat.SYNTHETIC, custom_loader)
        assert factory.get_loader(ModelFormat.SYNTHETIC) is custom_loader


class TestDefaultFactory:
    def test_get_default_factory(self):
        factory = get_default_factory()
        assert factory is not None

    def test_default_factory_singleton(self):
        factory1 = get_default_factory()
        factory2 = get_default_factory()
        assert factory1 is factory2

    def test_load_model_function(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        loaded = load_model(p)
        assert loaded.total_parameters == 256 * 512

    def test_load_model_metadata_function(self, tmp_path):
        arr = np.random.randn(256, 512).astype(np.float16)
        p = tmp_path / "model.npy"
        np.save(p, arr)

        metadata = load_model_metadata(p)
        assert metadata.total_parameters == 256 * 512