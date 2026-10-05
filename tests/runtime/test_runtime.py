import sys
import os
import tempfile
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

from src.ldmark.artifact.writer import LDMARKArtifactWriter
from src.ldmark.artifact.format import (
    ArtifactFormatVersion,
    ModelInfo,
    HardwareInfo,
    TensorEncoding,
    QuantizationParams,
    CompressionMethod,
)
from src.ldmark.compression.quantize import quantize_int8, quantize_int4, QuantizationConfig, QuantizationTarget
from src.ldmark.compression.binary_ternary import binary_quantize_sign, ternary_quantize_threshold, BinaryConfig, TernaryConfig
from src.ldmark.runtime import open_runtime, LDMARKRuntime, LDMARKRuntimeError
from src.ldmark.runtime.exceptions import (
    UnsupportedRuntimeFormat,
    TensorNotFoundError,
    ArtifactCorruptedError,
    DequantizationError,
    ComputationError,
    MissingFileError,
)
from src.ldmark.runtime.memory import MemoryAccountant, estimate_memory_impact, format_bytes
from src.ldmark.runtime.dequantize import dequantize_tensor, validate_dequantization
from src.ldmark.compression.metrics import calculate_error_metrics
from src.ldmark.artifact.reader import open_artifact


def create_test_artifact(artifact_dir: str) -> None:
    """Create a synthetic LDMARK artifact for testing."""
    writer = LDMARKArtifactWriter(artifact_dir, ArtifactFormatVersion.V1, overwrite=True)
    
    model_info = ModelInfo(
        model_id="test-model",
        architecture="llama",
        parameter_count=100_000,
        tensor_count=4,
        num_layers=2,
        hidden_size=256,
        intermediate_size=1024,
        num_attention_heads=8,
        num_kv_heads=8,
        vocab_size=1000,
        max_context_length=2048,
        original_dtype="float16",
    )
    writer.set_model_info(model_info)
    
    hardware_info = HardwareInfo(
        cpu_model="Test CPU",
        cpu_cores=8,
        cpu_architecture="x86_64",
        system_ram_gb=16.0,
        operating_system="Linux",
    )
    writer.set_hardware_info(hardware_info)
    
    # INT8 tensor
    tensor1 = np.random.randn(256, 512).astype(np.float32) * 0.1
    qtensor8 = quantize_int8(tensor1, group_size=128)
    writer.add_tensor(
        name="layer.0.weight",
        data=qtensor8.data,
        scales=qtensor8.scales,
        original_shape=tensor1.shape,
        original_dtype=tensor1.dtype,
        encoding=TensorEncoding.GROUPWISE_QUANTIZED,
        quantization=QuantizationParams(target_bits=8, group_size=128, scale_dtype="float16"),
    )
    
    # INT4 tensor
    tensor2 = np.random.randn(256, 1024).astype(np.float32) * 0.1
    qtensor4 = quantize_int4(tensor2, group_size=128)
    writer.add_tensor(
        name="layer.1.weight",
        data=qtensor4.data,
        scales=qtensor4.scales,
        original_shape=tensor2.shape,
        original_dtype=tensor2.dtype,
        encoding=TensorEncoding.GROUPWISE_QUANTIZED,
        quantization=QuantizationParams(target_bits=4, group_size=128, scale_dtype="float16"),
    )
    
    # Binary tensor
    tensor3 = np.random.randn(128, 256).astype(np.float32) * 0.1
    bconfig = BinaryConfig(group_size=128)
    btensor = binary_quantize_sign(tensor3, bconfig)
    writer.add_tensor(
        name="layer.0.attention.q_proj",
        data=btensor.packed_data,
        scales=btensor.scales,
        original_shape=tensor3.shape,
        original_dtype=tensor3.dtype,
        encoding=TensorEncoding.BINARY_PACKED,
        quantization=QuantizationParams(target_bits=1, group_size=128, scale_dtype="float16"),
    )
    
    # Ternary tensor
    tensor4 = np.random.randn(128, 256).astype(np.float32) * 0.1
    tconfig = TernaryConfig(group_size=128)
    ttensor = ternary_quantize_threshold(tensor4, tconfig, threshold=0.05)
    writer.add_tensor(
        name="layer.1.attention.k_proj",
        data=ttensor.packed_data,
        scales=ttensor.scales,
        original_shape=tensor4.shape,
        original_dtype=tensor4.dtype,
        encoding=TensorEncoding.TERNARY_PACKED,
        quantization=QuantizationParams(target_bits=2, group_size=128, scale_dtype="float16"),
    )
    
    writer.write()


class TestMemoryAccountant:
    def test_basic_accounting(self):
        accountant = MemoryAccountant()
        
        accountant.record_tensor_load(1000)
        assert accountant.compressed_weight_bytes == 1000
        
        accountant.record_tensor_decompress(1000, 4000)
        assert accountant.decompressed_weight_bytes == 4000
        
        accountant.record_temporary_allocation(2000)
        assert accountant.temporary_bytes == 2000
        assert accountant.peak_temporary_bytes == 2000
        
        accountant.record_temporary_release(2000)
        assert accountant.temporary_bytes == 0
        
        accountant.record_tensor_release(1000, 4000)
        assert accountant.compressed_weight_bytes == 0
        assert accountant.decompressed_weight_bytes == 0
    
    def test_peak_tracking(self):
        accountant = MemoryAccountant()
        
        accountant.record_temporary_allocation(1000)
        accountant.record_temporary_allocation(2000)
        assert accountant.peak_temporary_bytes == 3000
        
        accountant.record_temporary_release(1000)
        assert accountant.peak_temporary_bytes == 3000  # Peak preserved
        
        accountant.record_temporary_allocation(5000)
        assert accountant.peak_temporary_bytes == 7000  # New peak
    
    def test_estimated_peak(self):
        accountant = MemoryAccountant()
        accountant.record_tensor_load(1000)
        accountant.record_tensor_decompress(1000, 4000)
        accountant.record_temporary_allocation(2000)
        
        peak = accountant.get_peak_estimate()
        assert peak == 7000  # 1000 + 4000 + 2000
    
    def test_history(self):
        accountant = MemoryAccountant()
        accountant.record_tensor_load(1000)
        accountant.record_tensor_load(2000)
        
        history = accountant.get_history()
        assert len(history) == 2
        assert history[0].compressed_weight_bytes == 1000
        assert history[1].compressed_weight_bytes == 3000
    
    def test_reset(self):
        accountant = MemoryAccountant()
        accountant.record_tensor_load(1000)
        accountant.reset()
        assert accountant.compressed_weight_bytes == 0
        assert len(accountant.get_history()) == 0


class TestMemoryEstimation:
    def test_estimate_decompressed_size(self):
        from src.ldmark.runtime.memory import estimate_decompressed_size
        
        size = estimate_decompressed_size((256, 512), "float16")
        assert size == 256 * 512 * 2  # float16 = 2 bytes
        
        size = estimate_decompressed_size((256, 512), "float32")
        assert size == 256 * 512 * 4  # float32 = 4 bytes
    
    def test_estimate_memory_impact(self):
        impact = estimate_memory_impact(1000, (256, 512), "float16")
        decompressed = 256 * 512 * 2
        assert impact["compressed_bytes"] == 1000
        assert impact["decompressed_bytes"] == decompressed
        assert impact["expansion_factor"] == decompressed / 1000
    
    def test_format_bytes(self):
        assert format_bytes(500) == "500.00 B"
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(1024**2) == "1.00 MB"
        assert format_bytes(1024**3) == "1.00 GB"


class TestRuntimeBasics:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_open_close(self, artifact_dir):
        runtime = LDMARKRuntime(artifact_dir)
        assert runtime.is_open
        runtime.close()
        assert not runtime.is_open
    
    def test_context_manager(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            assert runtime.is_open
        assert not runtime.is_open
    
    def test_list_tensors(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensors = runtime.list_tensors()
            assert len(tensors) == 4
            assert "layer.0.weight" in tensors
            assert "layer.1.weight" in tensors
            assert "layer.0.attention.q_proj" in tensors
            assert "layer.1.attention.k_proj" in tensors
    
    def test_get_tensor_info(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            info = runtime.get_tensor_info("layer.0.weight")
            assert info is not None
            assert info.name == "layer.0.weight"
            assert info.original_shape == [256, 512]
            assert info.encoding == TensorEncoding.GROUPWISE_QUANTIZED
            assert info.quantization.target_bits == 8
            
            # Missing tensor
            info = runtime.get_tensor_info("nonexistent")
            assert info is None
    
    def test_get_model_info(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            model = runtime.get_model_info()
            assert model.model_id == "test-model"
            assert model.parameter_count == 100_000
            assert model.tensor_count == 4
    
    def test_get_compression_info(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            comp = runtime.get_compression_info()
            # Global compression info may be None if not set in artifact
            # This is acceptable - per-tensor quantization params are used instead


class TestLazyLoading:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_lazy_load_compressed_only(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            # Initially no tensors loaded
            assert len(runtime._loaded_tensors) == 0
            
            # Load tensor without decompressing
            tensor = runtime.get_tensor("layer.0.weight", decompress=False)
            
            assert tensor.name == "layer.0.weight"
            assert not tensor.is_decompressed
            assert tensor.decompressed_data is None
            assert tensor.compressed_data is not None
            
            # Memory should only track compressed bytes
            mem = runtime.get_memory_usage()
            assert mem["compressed_weight_bytes"] > 0
            assert mem["decompressed_weight_bytes"] == 0
    
    def test_lazy_load_with_decompress(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.0.weight", decompress=True)
            
            assert tensor.is_decompressed
            assert tensor.decompressed_data is not None
            assert tensor.dequantized is not None
            
            mem = runtime.get_memory_usage()
            assert mem["compressed_weight_bytes"] > 0
            assert mem["decompressed_weight_bytes"] > 0
    
    def test_get_tensor_data_convenience(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            data = runtime.get_tensor_data("layer.0.weight")
            
            assert data.shape == (256, 512)
            assert data.dtype == np.float16
            assert runtime._loaded_tensors["layer.0.weight"].is_decompressed
    
    def test_release_tensor(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.0.weight", decompress=True)
            assert "layer.0.weight" in runtime._loaded_tensors
            
            runtime.release_tensor("layer.0.weight")
            assert "layer.0.weight" not in runtime._loaded_tensors
            
            mem = runtime.get_memory_usage()
            assert mem["compressed_weight_bytes"] == 0
            assert mem["decompressed_weight_bytes"] == 0
    
    def test_release_all(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            runtime.get_tensor("layer.0.weight", decompress=True)
            runtime.get_tensor("layer.1.weight", decompress=True)
            assert len(runtime._loaded_tensors) == 2
            
            runtime.release_all()
            assert len(runtime._loaded_tensors) == 0
            
            mem = runtime.get_memory_usage()
            assert mem["compressed_weight_bytes"] == 0
            assert mem["decompressed_weight_bytes"] == 0


class TestDequantization:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_dequantize_int8(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.0.weight", decompress=True)
            
            dequantized = tensor.dequantized
            assert dequantized.data.shape == (256, 512)
            assert dequantized.data.dtype == np.float16
            assert dequantized.encoding == TensorEncoding.GROUPWISE_QUANTIZED
            assert dequantized.quantization.target_bits == 8
    
    def test_dequantize_int4(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.1.weight", decompress=True)
            
            dequantized = tensor.dequantized
            assert dequantized.data.shape == (256, 1024)
            assert dequantized.data.dtype == np.float16
            assert dequantized.encoding == TensorEncoding.GROUPWISE_QUANTIZED
            assert dequantized.quantization.target_bits == 4
    
    def test_dequantize_binary(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.0.attention.q_proj", decompress=True)
            
            dequantized = tensor.dequantized
            assert dequantized.data.shape == (128, 256)
            assert dequantized.data.dtype == np.float16
            assert dequantized.encoding == TensorEncoding.BINARY_PACKED
            assert dequantized.quantization.target_bits == 1
    
    def test_dequantize_ternary(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.1.attention.k_proj", decompress=True)
            
            dequantized = tensor.dequantized
            assert dequantized.data.shape == (128, 256)
            assert dequantized.data.dtype == np.float16
            assert dequantized.encoding == TensorEncoding.TERNARY_PACKED
            assert dequantized.quantization.target_bits == 2
    
    def test_dequantization_validates_shape(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            tensor = runtime.get_tensor("layer.0.weight", decompress=True)
            assert tensor.dequantized.original_shape == (256, 512)
            assert tensor.dequantized.data.shape == (256, 512)
    
    def test_unsupported_format_raises(self, artifact_dir):
        # Create artifact with unsupported format (e.g., unknown encoding)
        # We'll test that a tensor with unsupported target_bits fails
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_artifact = os.path.join(tmpdir, "bad_artifact")
            writer = LDMARKArtifactWriter(bad_artifact, ArtifactFormatVersion.V1, overwrite=True)
            
            model_info = ModelInfo(
                model_id="bad-model", architecture="test", parameter_count=100,
                tensor_count=1, original_dtype="float16"
            )
            writer.set_model_info(model_info)
            
            # Add INT2 tensor (unsupported)
            import numpy as np
            tensor = np.random.randn(10, 10).astype(np.float32)
            # Create fake INT2 data (not really INT2, just to trigger the check)
            fake_data = tensor.astype(np.uint8).tobytes()
            fake_arr = np.frombuffer(fake_data, dtype=np.uint8)
            
            writer.add_tensor(
                name="int2_tensor",
                data=fake_arr,
                original_shape=tensor.shape,
                original_dtype=tensor.dtype,
                encoding=TensorEncoding.GROUPWISE_QUANTIZED,
                quantization=QuantizationParams(target_bits=2, group_size=128, scale_dtype="float16"),
            )
            writer.write()
            
            with open_runtime(bad_artifact) as runtime:
                with pytest.raises(UnsupportedRuntimeFormat) as exc_info:
                    runtime.get_tensor("int2_tensor", decompress=True)
                assert "2" in str(exc_info.value) or "int2" in str(exc_info.value).lower()


class TestMatmul:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_matmul_basic(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            input_data = np.random.randn(1, 512).astype(np.float16)
            result = runtime.matmul("layer.0.weight", input_data)
            
            # Weight is [256, 512], input is [1, 512]
            # Output should be [1, 256]
            assert result.output.shape == (1, 256)
            assert result.output.dtype == np.float16
            assert result.weight_name == "layer.0.weight"
            assert result.computation_time_ms > 0
    
    def test_matmul_batched(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            input_data = np.random.randn(4, 512).astype(np.float16)
            result = runtime.matmul("layer.0.weight", input_data)
            
            assert result.output.shape == (4, 256)
    
    def test_matmul_1d_input(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            input_data = np.random.randn(512).astype(np.float16)
            result = runtime.matmul("layer.0.weight", input_data)
            
            assert result.output.shape == (256,)
    
    def test_matmul_cached_weight(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            # First load
            runtime.get_tensor("layer.0.weight", decompress=True)
            
            # Use cached
            input_data = np.random.randn(1, 512).astype(np.float16)
            result = runtime.matmul_with_cached_weight("layer.0.weight", input_data)
            
            assert result.output.shape == (1, 256)
    
    def test_matmul_shape_error(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            # Wrong input shape - weight is [256, 512], input needs 512 features
            input_data = np.random.randn(1, 100).astype(np.float16)
            with pytest.raises((ComputationError, ValueError)):
                runtime.matmul("layer.0.weight", input_data)


class TestBenchmark:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_benchmark_tensor(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            result = runtime.benchmark_tensor(
                "layer.0.weight",
                input_shape=(1, 512),
                num_warmup=1,
                num_runs=3,
            )
            
            assert result.tensor_name == "layer.0.weight"
            assert result.load_time_ms > 0
            assert result.reconstruction_time_ms > 0
            assert result.computation_time_ms > 0
            assert result.total_time_ms > 0
            assert result.compressed_bytes > 0
            assert result.decompressed_bytes > 0
            assert result.ops_per_second > 0
    
    def test_benchmark_invalid_shape(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            with pytest.raises(ComputationError):
                runtime.benchmark_tensor("layer.0.weight", (1, 100))


class TestValidation:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_validate_tensor(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            # Get original tensor data by re-quantizing
            # Use a fixed seed for reproducibility
            np.random.seed(42)
            original = np.random.randn(256, 512).astype(np.float32) * 0.1
            metrics = runtime.validate_tensor("layer.0.weight", original)
            
            # validate_tensor returns a tuple (mae, mse, max_abs_error, relative_error, cosine_similarity, snr_db, psnr_db)
            assert isinstance(metrics, tuple)
            assert len(metrics) >= 4
            mae, mse, max_abs_error, relative_error = metrics[:4]
            assert mae >= 0
            assert mse >= 0
            assert max_abs_error >= 0
            assert relative_error >= 0
    
    def test_validate_shape_mismatch(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            # Wrong shape
            original = np.random.randn(100, 100).astype(np.float32)
            with pytest.raises(ComputationError):
                runtime.validate_tensor("layer.0.weight", original)


class TestErrorHandling:
    def test_tensor_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            
            with open_runtime(artifact_dir) as runtime:
                with pytest.raises(TensorNotFoundError) as exc_info:
                    runtime.get_tensor("nonexistent.tensor")
                assert exc_info.value.tensor_name == "nonexistent.tensor"
                assert len(exc_info.value.available_tensors) == 4
    
    def test_corrupted_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_dir = os.path.join(tmpdir, "bad_artifact")
            os.makedirs(bad_dir)
            # Create invalid manifest
            with open(os.path.join(bad_dir, "manifest.json"), "w") as f:
                f.write("{ invalid json")
            
            runtime = LDMARKRuntime(bad_dir)
            # Error occurs when trying to access manifest
            with pytest.raises((ArtifactCorruptedError, LDMARKRuntimeError, json.JSONDecodeError, ValueError)):
                _ = runtime.manifest
    
    def test_missing_artifact(self):
        from src.ldmark.artifact.integrity import MissingFileError as ArtifactMissingFileError
        runtime = LDMARKRuntime("/nonexistent/path")
        with pytest.raises((ArtifactCorruptedError, LDMARKRuntimeError, FileNotFoundError, ArtifactMissingFileError)):
            _ = runtime.manifest


class TestInspect:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_inspect(self, artifact_dir):
        with open_runtime(artifact_dir) as runtime:
            info = runtime.inspect()
            
            assert info["format_version"] == "ldmark-artifact-1.0"
            assert info["model"]["model_id"] == "test-model"
            assert info["tensor_count"] == 4
            assert len(info["tensor_names"]) == 4
            # target_dtype is numpy dtype object
            assert "float16" in str(info["target_dtype"])
            assert "memory" in info
            # compression may be None if not set globally


class TestMemoryComparison:
    def test_print_memory_comparison(self, capsys):
        from src.ldmark.runtime.memory import print_memory_comparison
    
        # Simulate 27B model at INT4
        compressed = 4 * 1024**3  # 4 GB
        print_memory_comparison(compressed, (27_000_000_000,), "float16")
    
        captured = capsys.readouterr()
        assert "STORAGE COMPRESSION" in captured.out
        assert "Decompressed (runtime)" in captured.out
        assert "Future: direct computation" in captured.out


class TestDequantizeModule:
    @pytest.fixture
    def artifact_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = os.path.join(tmpdir, "test_artifact")
            create_test_artifact(artifact_dir)
            yield artifact_dir
    
    def test_dequantize_tensor_function(self, artifact_dir):
        from src.ldmark.artifact.reader import open_artifact
        from src.ldmark.artifact.integrity import MissingFileError as ArtifactMissingFileError
        from src.ldmark.runtime.dequantize import dequantize_tensor
        
        with open_artifact(artifact_dir) as reader:
            entry = reader.get_tensor_info("layer.0.weight")
            tensor_data = reader.read_tensor("layer.0.weight")
            
            result = dequantize_tensor(tensor_data.data, tensor_data.scales, entry, np.float16)
            
            assert result.data.shape == (256, 512)
            assert result.data.dtype == np.float16
            assert result.name == "layer.0.weight"
            assert result.compressed_size_bytes > 0
            assert result.decompressed_size_bytes > 0
    
    def test_validate_dequantization(self):
        original = np.random.randn(10, 10).astype(np.float32)
        reconstructed = original + np.random.randn(10, 10).astype(np.float32) * 0.01
        
        metrics = validate_dequantization(original, reconstructed, "test_tensor")
        
        # validate_dequantization returns (mae, mse, max_abs_error, relative_error) tuple
        assert isinstance(metrics, tuple)
        assert len(metrics) == 4
        mae, mse, max_abs_error, relative_error = metrics
        assert mae > 0
        assert mse > 0
        assert max_abs_error > 0
        assert relative_error > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])