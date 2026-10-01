import pytest
from src.ldmark.file_io.metadata import (
    TensorDtype,
    TensorMetadata,
    ModelMetadata,
)
from src.ldmark.file_io.formats import ModelFormat


class TestTensorDtype:
    def test_dtype_values(self):
        assert TensorDtype.FLOAT32.value == "float32"
        assert TensorDtype.FLOAT16.value == "float16"
        assert TensorDtype.BFLOAT16.value == "bfloat16"
        assert TensorDtype.INT8.value == "int8"
        assert TensorDtype.UNKNOWN.value == "unknown"


class TestTensorMetadata:
    def test_creation(self):
        tm = TensorMetadata(
            name="weight",
            shape=[256, 512],
            dtype=TensorDtype.FLOAT16,
            num_elements=256 * 512,
            raw_bytes=256 * 512 * 2,
        )
        assert tm.name == "weight"
        assert tm.shape == [256, 512]
        assert tm.dtype == TensorDtype.FLOAT16
        assert tm.num_elements == 256 * 512
        assert tm.raw_bytes == 256 * 512 * 2

    def test_to_dict(self):
        tm = TensorMetadata(
            name="weight",
            shape=[256, 512],
            dtype=TensorDtype.FLOAT16,
            num_elements=256 * 512,
            raw_bytes=256 * 512 * 2,
        )
        d = tm.to_dict()
        assert d["name"] == "weight"
        assert d["shape"] == [256, 512]
        assert d["dtype"] == "float16"
        assert d["num_elements"] == 256 * 512
        assert d["raw_bytes"] == 256 * 512 * 2


class TestModelMetadata:
    def test_creation(self):
        metadata = ModelMetadata(
            model_id="test-model",
            source_path="/path/to/model",
            source_format=ModelFormat.NUMPY,
        )
        assert metadata.model_id == "test-model"
        assert metadata.source_path == "/path/to/model"
        assert metadata.source_format == ModelFormat.NUMPY
        assert metadata.architecture == "unknown"
        assert metadata.total_parameters == 0
        assert metadata.tensor_count == 0

    def test_to_dict(self):
        metadata = ModelMetadata(
            model_id="test-model",
            source_path="/path/to/model",
            source_format=ModelFormat.NUMPY,
            architecture="llama",
            num_layers=32,
            hidden_size=4096,
        )
        d = metadata.to_dict()
        assert d["model_id"] == "test-model"
        assert d["architecture"] == "llama"
        assert d["num_layers"] == 32
        assert d["hidden_size"] == 4096

    def test_get_architecture_summary(self):
        metadata = ModelMetadata(
            model_id="test-model",
            source_path="/path/to/model",
            source_format=ModelFormat.NUMPY,
            architecture="llama",
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_kv_heads=8,
            vocab_size=32000,
        )
        summary = metadata.get_architecture_summary()
        assert summary["architecture"] == "llama"
        assert summary["num_layers"] == 32
        assert summary["hidden_size"] == 4096
        assert summary["num_kv_heads"] == 8

    def test_with_tensors(self):
        tm = TensorMetadata(
            name="weight",
            shape=[256, 512],
            dtype=TensorDtype.FLOAT16,
            num_elements=256 * 512,
            raw_bytes=256 * 512 * 2,
        )
        metadata = ModelMetadata(
            model_id="test-model",
            source_path="/path/to/model",
            source_format=ModelFormat.NUMPY,
            tensors=[tm],
            total_parameters=256 * 512,
            tensor_count=1,
        )
        assert metadata.tensor_count == 1
        assert metadata.total_parameters == 256 * 512
        assert len(metadata.tensors) == 1