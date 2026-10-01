import pytest
import numpy as np
from src.ldmark.analysis.tensor import (
    create_tensor_info,
    analyze_numpy_dict,
    analyze_torch_state_dict,
    calculate_tensor_statistics,
    infer_dtype_from_name,
    NUMPY_TO_DTYPE,
)
from src.ldmark.analysis.models import DType, TensorTrainableStatus


class TestInferDtypeFromName:
    def test_torch_dtypes(self):
        assert infer_dtype_from_name("weight", "float32") == DType.FP32
        assert infer_dtype_from_name("weight", "float16") == DType.FP16
        assert infer_dtype_from_name("weight", "bfloat16") == DType.BF16
        assert infer_dtype_from_name("weight", "int8") == DType.INT8
        assert infer_dtype_from_name("weight", "int4") == DType.INT4

    def test_numpy_dtypes(self):
        assert infer_dtype_from_name("weight", "float32") == DType.FP32
        assert infer_dtype_from_name("weight", "float16") == DType.FP16

    def test_fallback(self):
        assert infer_dtype_from_name("weight", "unknown") == DType.FP16


class TestCreateTensorInfo:
    def test_from_shape_and_dtype_enum(self):
        tensor = create_tensor_info("test.weight", [256, 512], DType.FP16)
        assert tensor.name == "test.weight"
        assert tensor.shape == [256, 512]
        assert tensor.dtype == DType.FP16
        assert tensor.num_elements == 256 * 512

    def test_from_shape_and_dtype_string(self):
        tensor = create_tensor_info("test.weight", [256, 512], "float16")
        assert tensor.dtype == DType.FP16

    def test_from_shape_and_numpy_dtype(self):
        tensor = create_tensor_info("test.weight", [256, 512], np.float16)
        assert tensor.dtype == DType.FP16

    def test_trainable_status(self):
        tensor = create_tensor_info("test.weight", [256, 512], DType.FP16, TensorTrainableStatus.TRAINABLE)
        assert tensor.trainable == TensorTrainableStatus.TRAINABLE


class TestAnalyzeNumpyDict:
    def test_basic_analysis(self):
        tensors = {
            "layer1.weight": np.random.randn(256, 512).astype(np.float16),
            "layer2.weight": np.random.randn(512, 256).astype(np.float32),
            "bias": np.random.randn(256).astype(np.float16),
        }
        results = analyze_numpy_dict(tensors)
        assert len(results) == 3
        assert results[0].dtype == DType.FP16
        assert results[1].dtype == DType.FP32
        assert results[2].dtype == DType.FP16

    def test_trainable_names(self):
        tensors = {
            "layer1.weight": np.random.randn(256, 512).astype(np.float16),
            "layer2.weight": np.random.randn(512, 256).astype(np.float16),
        }
        results = analyze_numpy_dict(tensors, trainable_names={"layer1.weight"})
        assert results[0].trainable == TensorTrainableStatus.TRAINABLE
        assert results[1].trainable == TensorTrainableStatus.FROZEN


class TestCalculateTensorStatistics:
    def test_empty_list(self):
        stats = calculate_tensor_statistics([])
        assert stats["total_tensors"] == 0
        assert stats["total_elements"] == 0
        assert stats["total_raw_bytes"] == 0

    def test_with_tensors(self):
        from src.ldmark.analysis.tensor import create_tensor_info
        tensors = [
            create_tensor_info("t1", [256, 512], DType.FP16),
            create_tensor_info("t2", [512, 256], DType.FP16),
            create_tensor_info("t3", [1024], DType.FP32),
        ]
        stats = calculate_tensor_statistics(tensors)
        assert stats["total_tensors"] == 3
        assert stats["total_elements"] == 256 * 512 * 2 + 1024
        assert "float16" in stats["by_dtype"]
        assert "float32" in stats["by_dtype"]
        assert stats["shape_stats"]["unique_shapes"] == 3