import pytest
from src.ldmark.analysis.models import (
    DType,
    TensorTrainableStatus,
    TensorInfo,
    ParameterCounts,
    ArchitectureInfo,
    StorageEstimate,
    CompressionEstimate,
    RuntimeMemoryEstimate,
    ModelAnalysis,
)


class TestDType:
    def test_bits_property(self):
        assert DType.FP32.bits == 32
        assert DType.FP16.bits == 16
        assert DType.BF16.bits == 16
        assert DType.INT8.bits == 8
        assert DType.INT4.bits == 4
        assert DType.FP8.bits == 8
        assert DType.FP4.bits == 4
        assert DType.BIT1.bits == 1

    def test_bytes_per_element_property(self):
        assert DType.FP32.bytes_per_element == 4.0
        assert DType.FP16.bytes_per_element == 2.0
        assert DType.BF16.bytes_per_element == 2.0
        assert DType.INT8.bytes_per_element == 1.0
        assert DType.INT4.bytes_per_element == 0.5
        assert DType.FP8.bytes_per_element == 1.0
        assert DType.FP4.bytes_per_element == 0.5
        assert DType.BIT1.bytes_per_element == 0.125


class TestTensorInfo:
    def test_from_shape_and_dtype(self):
        tensor = TensorInfo.from_shape_and_dtype("test.weight", [256, 512], DType.FP16)
        assert tensor.name == "test.weight"
        assert tensor.shape == [256, 512]
        assert tensor.dtype == DType.FP16
        assert tensor.num_elements == 256 * 512
        assert tensor.estimated_raw_storage_bytes == 256 * 512 * 2

    def test_to_dict(self):
        tensor = TensorInfo.from_shape_and_dtype("test.weight", [256, 512], DType.FP16)
        d = tensor.to_dict()
        assert d["name"] == "test.weight"
        assert d["shape"] == [256, 512]
        assert d["dtype"] == "float16"
        assert d["num_elements"] == 256 * 512


class TestParameterCounts:
    def test_add_tensor(self):
        counts = ParameterCounts()
        counts.add_tensor("layer1.weight", 1000, DType.FP16, TensorTrainableStatus.TRAINABLE)
        counts.add_tensor("layer2.weight", 2000, DType.FP32, TensorTrainableStatus.FROZEN)
        counts.add_tensor("layer3.bias", 500, DType.FP16, TensorTrainableStatus.UNKNOWN)

        assert counts.total == 3500
        assert counts.by_dtype[DType.FP16] == 1500
        assert counts.by_dtype[DType.FP32] == 2000
        assert counts.trainable == 1000
        assert counts.non_trainable == 2000
        assert counts.unknown_trainable == 500

    def test_to_dict(self):
        counts = ParameterCounts()
        counts.add_tensor("layer1.weight", 1000, DType.FP16, TensorTrainableStatus.TRAINABLE)
        d = counts.to_dict()
        assert d["total"] == 1000
        assert d["by_dtype"]["float16"] == 1000
        assert d["trainable"] == 1000


class TestArchitectureInfo:
    def test_to_dict_with_none_values(self):
        arch = ArchitectureInfo(
            model_architecture="llama",
            num_layers=32,
            hidden_size=4096,
            vocab_size=32000,
        )
        d = arch.to_dict()
        assert d["model_architecture"] == "llama"
        assert d["num_layers"] == 32
        assert d["hidden_size"] == 4096
        assert d["vocab_size"] == 32000
        assert "num_kv_heads" not in d

    def test_to_dict_with_additional_metadata(self):
        arch = ArchitectureInfo(
            model_architecture="llama",
            additional_metadata={"custom_field": "value", "another": 123},
        )
        d = arch.to_dict()
        assert d["custom_field"] == "value"
        assert d["another"] == 123


class TestStorageEstimate:
    def test_from_parameter_count(self):
        est = StorageEstimate.from_parameter_count(1_000_000_000, DType.FP16)
        assert est.dtype == DType.FP16
        assert est.estimated_bytes == 2_000_000_000
        assert est.estimated_gb == pytest.approx(1.8626, rel=0.01)

    def test_to_dict(self):
        est = StorageEstimate.from_parameter_count(1_000_000_000, DType.FP16)
        d = est.to_dict()
        assert d["dtype"] == "float16"
        assert d["estimated_bytes"] == 2_000_000_000
        assert d["is_theoretical"] is True


class TestCompressionEstimate:
    def test_from_parameters(self):
        est = CompressionEstimate.from_parameters(
            param_count=1_000_000_000,
            format_name="INT4",
            effective_bits_per_weight=4.0,
            baseline_dtype=DType.FP16,
        )
        assert est.format_name == "INT4"
        assert est.effective_bits_per_weight == 4.0
        assert est.estimated_bytes == 500_000_000
        assert est.compression_ratio == 4.0

    def test_from_parameters_with_group_scale(self):
        est = CompressionEstimate.from_parameters(
            param_count=1_000_000_000,
            format_name="Bonsai Q1_0_g128",
            effective_bits_per_weight=1.125,
            baseline_dtype=DType.FP16,
            group_size=128,
            scale_dtype=DType.FP16,
        )
        assert est.group_size == 128
        assert est.scale_dtype == DType.FP16
        assert est.estimated_bytes > 125_000_000


class TestRuntimeMemoryEstimate:
    def test_estimate(self):
        est = RuntimeMemoryEstimate.estimate(
            param_count=7_000_000_000,
            weight_dtype=DType.FP16,
            context_length=2048,
            hidden_size=4096,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
        )
        assert est.weights_gb > 0
        assert est.kv_cache_gb > 0
        assert est.activations_gb > 0
        assert est.total_gb == est.weights_gb + est.kv_cache_gb + est.activations_gb + est.overhead_gb


class TestModelAnalysis:
    def test_to_json(self):
        analysis = ModelAnalysis(
            model_id="test-model",
            source_path="/path/to/model",
            parameter_counts=ParameterCounts(),
            tensors=[],
            architecture=ArchitectureInfo(),
            storage_estimates=[],
            compression_estimates=[],
        )
        json_str = analysis.to_json()
        assert "test-model" in json_str
        assert "parameter_counts" in json_str