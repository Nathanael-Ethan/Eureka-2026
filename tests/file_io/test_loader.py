import pytest
import numpy as np
from src.ldmark.file_io.loader import (
    ModelLoader,
    BaseModelLoader,
    ModelLoaderError,
    UnsupportedFormatError,
    CorruptFileError,
    MissingDependencyError,
    MalformedStateDictError,
)
from src.ldmark.file_io.formats import ModelFormat, ModelSource


class TestModelLoaderErrors:
    def test_error_hierarchy(self):
        assert issubclass(UnsupportedFormatError, ModelLoaderError)
        assert issubclass(CorruptFileError, ModelLoaderError)
        assert issubclass(MissingDependencyError, ModelLoaderError)
        assert issubclass(MalformedStateDictError, ModelLoaderError)

    def test_unsupported_format_error(self):
        err = UnsupportedFormatError("test format")
        assert str(err) == "test format"

    def test_corrupt_file_error(self):
        err = CorruptFileError("corrupt")
        assert str(err) == "corrupt"

    def test_missing_dependency_error(self):
        err = MissingDependencyError("torch required")
        assert str(err) == "torch required"

    def test_malformed_state_dict_error(self):
        err = MalformedStateDictError("bad state dict")
        assert str(err) == "bad state dict"


class TestBaseModelLoader:
    def test_extract_architecture_from_config(self):
        loader = BaseModelLoader()
        config = {
            "architectures": ["LlamaForCausalLM"],
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        }
        result = loader._extract_architecture_from_config(config)
        assert result["architecture"] == "LlamaForCausalLM"
        assert result["num_layers"] == 32
        assert result["hidden_size"] == 4096
        assert result["num_kv_heads"] == 8

    def test_extract_architecture_unknown(self):
        loader = BaseModelLoader()
        result = loader._extract_architecture_from_config({})
        assert result["architecture"] == "unknown"

    def test_detect_attention_type_mha(self):
        loader = BaseModelLoader()
        config = {"num_attention_heads": 32, "num_key_value_heads": 32}
        assert loader._detect_attention_type(config) == "mha"

    def test_detect_attention_type_mqa(self):
        loader = BaseModelLoader()
        config = {"num_attention_heads": 32, "num_key_value_heads": 1}
        assert loader._detect_attention_type(config) == "mqa"

    def test_detect_attention_type_gqa(self):
        loader = BaseModelLoader()
        config = {"num_attention_heads": 32, "num_key_value_heads": 8}
        assert loader._detect_attention_type(config) == "gqa"

    def test_detect_attention_type_none(self):
        loader = BaseModelLoader()
        assert loader._detect_attention_type({}) is None

    def test_numpy_dtype_conversion(self):
        loader = BaseModelLoader()
        assert loader._numpy_dtype_to_tensor_dtype(np.dtype("float32")).value == "float32"
        assert loader._numpy_dtype_to_tensor_dtype(np.dtype("float16")).value == "float16"
        assert loader._numpy_dtype_to_tensor_dtype(np.dtype("int8")).value == "int8"
        assert loader._numpy_dtype_to_tensor_dtype(np.dtype("uint8")).value == "uint8"