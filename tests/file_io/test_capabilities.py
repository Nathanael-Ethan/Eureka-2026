import pytest
from src.ldmark.file_io.capabilities import (
    is_torch_available,
    is_safetensors_available,
    is_numpy_available,
    get_torch_version,
    get_safetensors_version,
    get_capabilities,
)


class TestCapabilityDetection:
    def test_is_torch_available_returns_bool(self):
        result = is_torch_available()
        assert isinstance(result, bool)

    def test_is_safetensors_available_returns_bool(self):
        result = is_safetensors_available()
        assert isinstance(result, bool)

    def test_is_numpy_available_returns_bool(self):
        result = is_numpy_available()
        assert isinstance(result, bool)

    def test_numpy_should_be_available(self):
        assert is_numpy_available() is True

    def test_get_torch_version(self):
        version = get_torch_version()
        if is_torch_available():
            assert version is not None
            assert isinstance(version, str)
        else:
            assert version is None

    def test_get_safetensors_version(self):
        version = get_safetensors_version()
        if is_safetensors_available():
            assert version is not None
            assert isinstance(version, str)
        else:
            assert version is None

    def test_get_capabilities(self):
        caps = get_capabilities()
        assert "torch" in caps
        assert "safetensors" in caps
        assert "numpy" in caps
        assert "available" in caps["torch"]
        assert "available" in caps["safetensors"]
        assert "available" in caps["numpy"]