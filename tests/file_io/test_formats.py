import pytest
import json
import numpy as np
from pathlib import Path
from src.ldmark.file_io.formats import (
    ModelFormat,
    ModelSource,
    detect_format_from_path,
    detect_format_from_content,
)


class TestModelFormat:
    def test_format_values(self):
        assert ModelFormat.NUMPY.value == "numpy"
        assert ModelFormat.PYTORCH.value == "pytorch"
        assert ModelFormat.SAFETENSORS.value == "safetensors"
        assert ModelFormat.JSON_CONFIG.value == "json_config"
        assert ModelFormat.SYNTHETIC.value == "synthetic"
        assert ModelFormat.UNKNOWN.value == "unknown"


class TestModelSource:
    def test_model_source_creation(self):
        source = ModelSource("/path/to/model", ModelFormat.NUMPY)
        assert source.path == Path("/path/to/model")
        assert source.format == ModelFormat.NUMPY
        assert source.is_directory is False

    def test_model_source_name(self):
        source = ModelSource("/path/to/model.npy", ModelFormat.NUMPY)
        assert source.name == "model.npy"
        assert source.suffix == ".npy"

    def test_model_source_repr(self):
        source = ModelSource("/path/to/model", ModelFormat.NUMPY)
        assert "model" in repr(source)
        assert "numpy" in repr(source)


class TestDetectFormatFromFile:
    def test_detect_npy(self, tmp_path):
        p = tmp_path / "model.npy"
        np.save(p, np.array([1, 2, 3]))
        source = detect_format_from_path(p)
        assert source.format == ModelFormat.NUMPY

    def test_detect_npz(self, tmp_path):
        p = tmp_path / "model.npz"
        np.savez(p, arr=np.array([1, 2, 3]))
        source = detect_format_from_path(p)
        assert source.format == ModelFormat.NUMPY

    def test_detect_pytorch(self, tmp_path):
        p = tmp_path / "model.bin"
        p.write_bytes(b"fake")
        source = detect_format_from_path(p)
        assert source.format == ModelFormat.PYTORCH

    def test_detect_safetensors(self, tmp_path):
        p = tmp_path / "model.safetensors"
        p.write_bytes(b"fake")
        source = detect_format_from_path(p)
        assert source.format == ModelFormat.SAFETENSORS

    def test_detect_json_config(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"model_type": "llama"}))
        source = detect_format_from_path(p)
        assert source.format == ModelFormat.JSON_CONFIG

    def test_detect_unknown(self, tmp_path):
        p = tmp_path / "model.unknown"
        p.write_bytes(b"fake")
        source = detect_format_from_path(p)
        assert source.format == ModelFormat.UNKNOWN

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            detect_format_from_path("/nonexistent/path")


class TestDetectFormatFromDirectory:
    def test_detect_directory_with_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.JSON_CONFIG
        assert source.is_directory is True

    def test_detect_directory_with_safetensors(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.SAFETENSORS

    def test_detect_directory_with_pytorch(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        (tmp_path / "pytorch_model.bin").write_bytes(b"fake")
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.PYTORCH

    def test_detect_directory_with_numpy(self, tmp_path):
        np.save(tmp_path / "weights.npy", np.array([1, 2, 3]))
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.NUMPY

    def test_detect_directory_with_config_and_numpy_is_numpy(self, tmp_path):
        """config.json next to NumPy weights should not mask the tensors —
        the NUMPY loader reads the config as a sidecar."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        np.save(tmp_path / "weights.npy", np.array([1, 2, 3]))
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.NUMPY
        assert source.is_directory is True

    def test_detect_directory_with_broken_config_json_falls_to_json(self, tmp_path):
        """An unreadable config.json with no tensor files still resolves to
        JSON_CONFIG via the generic .json rule."""
        (tmp_path / "config.json").write_text("{not valid json")
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.JSON_CONFIG
        assert source.is_directory is True

    def test_detect_empty_directory(self, tmp_path):
        source = detect_format_from_path(tmp_path)
        assert source.format == ModelFormat.UNKNOWN


class TestDetectFormatFromContent:
    def test_detect_json_content(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text(json.dumps({"model_type": "llama"}))
        source = detect_format_from_content(p)
        assert source.format == ModelFormat.JSON_CONFIG

    def test_detect_invalid_json(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("not valid json")
        source = detect_format_from_content(p)
        assert source.format == ModelFormat.UNKNOWN