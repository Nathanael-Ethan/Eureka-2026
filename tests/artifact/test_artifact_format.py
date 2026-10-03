"""
Tests for LDMARK Artifact Format
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ldmark.artifact import (
    LDMARKArtifactWriter,
    LDMARKArtifactReader,
    open_artifact,
    inspect_artifact,
    verify_artifact,
    Manifest,
    TensorIndexEntry,
    CompressionInfo,
    ModelInfo,
    HardwareInfo,
    CompilationInfo,
    ValidationInfo,
    StorageAccounting,
    TensorEncoding,
    QuantizationParams,
    CompressionMethod,
    ArtifactFormatVersion,
    IntegrityError,
    ChecksumMismatchError,
    SizeMismatchError,
    MissingFileError,
    VersionMismatchError,
    quick_verify,
)


class TestTensorIndexEntry:
    """Tests for TensorIndexEntry."""

    def test_tensor_index_entry_creation(self):
        entry = TensorIndexEntry(
            name="test.weight",
            original_shape=[128, 256],
            original_dtype="float16",
            original_size_bytes=65536,
            encoding=TensorEncoding.GROUPWISE_QUANTIZED,
            quantization=QuantizationParams(target_bits=4, group_size=128),
            filename="test_weight.bin",
            byte_length=16384,
            scale_filename="test_weight.scales.bin",
            scale_byte_length=512,
            num_scales=128,
            sha256="abc123",
            scale_sha256="def456",
        )

        assert entry.name == "test.weight"
        assert entry.original_shape == [128, 256]
        assert entry.bits_per_weight == 0.0

    def test_tensor_index_entry_serialization(self):
        entry = TensorIndexEntry(
            name="test.weight",
            original_shape=[128, 256],
            original_dtype="float16",
            original_size_bytes=65536,
            encoding=TensorEncoding.GROUPWISE_QUANTIZED,
            quantization=QuantizationParams(target_bits=4, group_size=128),
            filename="test_weight.bin",
            byte_length=16384,
            scale_filename="test_weight.scales.bin",
            scale_byte_length=512,
            num_scales=128,
        )

        data = entry.to_dict()
        assert data["name"] == "test.weight"
        assert data["encoding"] == "groupwise_quantized"
        assert data["quantization"]["target_bits"] == 4

        restored = TensorIndexEntry.from_dict(data)
        assert restored.name == entry.name
        assert restored.quantization.target_bits == entry.quantization.target_bits


class TestManifest:
    """Tests for Manifest."""

    def test_manifest_creation(self):
        model = ModelInfo(
            model_id="test_model",
            architecture="llama",
            parameter_count=1000000,
            tensor_count=10,
        )

        compression = CompressionInfo(
            method=CompressionMethod.INT4,
            target_bits=4,
            group_size=128,
        )

        compilation = CompilationInfo(
            ldmark_version="0.1.0",
            format_version="ldmark-artifact-1.0",
            created_at="2024-01-01T00:00:00",
        )

        manifest = Manifest(
            format_version=ArtifactFormatVersion.V1,
            model=model,
            compression=compression,
            compilation=compilation,
        )

        assert manifest.magic == "LDMARK"
        assert manifest.format_version == ArtifactFormatVersion.V1
        assert manifest.model is not None
        assert manifest.compression is not None

    def test_manifest_checksum(self):
        manifest = Manifest(
            format_version=ArtifactFormatVersion.V1,
            model=ModelInfo(model_id="test", architecture="test", parameter_count=100, tensor_count=1),
            compression=CompressionInfo(method=CompressionMethod.INT8, target_bits=8, group_size=128),
        )

        assert manifest.manifest_sha256 == ""

        sha256 = manifest.compute_sha256()
        manifest.manifest_sha256 = sha256
        assert manifest.verify_integrity() is True

        manifest.model.model_id = "tampered"
        assert manifest.verify_integrity() is False

    def test_manifest_serialization(self):
        manifest = Manifest(
            format_version=ArtifactFormatVersion.V1,
            model=ModelInfo(model_id="test", architecture="test", parameter_count=100, tensor_count=1),
            compression=CompressionInfo(method=CompressionMethod.INT8, target_bits=8, group_size=128),
            tensors=[
                TensorIndexEntry(
                    name="weight",
                    original_shape=[10, 10],
                    original_dtype="float16",
                    original_size_bytes=200,
                    encoding=TensorEncoding.GROUPWISE_QUANTIZED,
                    quantization=QuantizationParams(target_bits=8, group_size=128),
                    filename="weight.bin",
                    byte_length=100,
                    num_scales=1,
                )
            ],
        )

        json_str = manifest.to_json()
        parsed = json.loads(json_str)
        assert parsed["magic"] == "LDMARK"
        assert parsed["format_version"] == "ldmark-artifact-1.0"
        assert len(parsed["tensors"]) == 1

        restored = Manifest.from_dict(parsed)
        assert restored.model.model_id == "test"
        assert len(restored.tensors) == 1

    def test_get_tensor_index(self):
        manifest = Manifest(
            format_version=ArtifactFormatVersion.V1,
            model=ModelInfo(model_id="test", architecture="test", parameter_count=100, tensor_count=2),
            compression=CompressionInfo(method=CompressionMethod.INT8, target_bits=8, group_size=128),
            tensors=[
                TensorIndexEntry(
                    name="weight1",
                    original_shape=[10, 10],
                    original_dtype="float16",
                    original_size_bytes=200,
                    encoding=TensorEncoding.GROUPWISE_QUANTIZED,
                    quantization=QuantizationParams(target_bits=8, group_size=128),
                    filename="weight1.bin",
                    byte_length=100,
                ),
                TensorIndexEntry(
                    name="weight2",
                    original_shape=[10, 10],
                    original_dtype="float16",
                    original_size_bytes=200,
                    encoding=TensorEncoding.GROUPWISE_QUANTIZED,
                    quantization=QuantizationParams(target_bits=8, group_size=128),
                    filename="weight2.bin",
                    byte_length=100,
                ),
            ],
        )

        index = manifest.get_tensor_index()
        assert "weight1" in index
        assert "weight2" in index
        assert index["weight1"].byte_length == 100


class TestArtifactWriter:
    """Tests for LDMARKArtifactWriter."""

    def test_write_empty_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="test_model",
                architecture="test",
                parameter_count=100,
                tensor_count=0,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT8,
                target_bits=8,
                group_size=128,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
            ))

            artifact = writer.write()

            assert Path(artifact.artifact_path).resolve() == artifact_dir.resolve()
            assert artifact.manifest.model.model_id == "test_model"
            assert len(artifact.manifest.tensors) == 0
            assert (artifact_dir / "manifest.json").exists()
            assert (artifact_dir / "tensors").exists()

    def test_write_artifact_with_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="test_model",
                architecture="llama",
                parameter_count=65536,
                tensor_count=2,
                hidden_size=256,
                num_layers=2,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT4,
                target_bits=4,
                group_size=128,
                scale_dtype="float16",
                symmetric=True,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
                optimization_strategy="balanced",
            ))

            data1 = np.random.randint(0, 255, size=8192, dtype=np.uint8)
            scales1 = np.random.randn(64).astype(np.float16)

            data2 = np.random.randint(0, 255, size=4096, dtype=np.uint8)
            scales2 = np.random.randn(32).astype(np.float16)

            writer.add_tensor(
                name="layer1.weight",
                data=data1,
                scales=scales1,
                original_shape=(256, 256),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=4, group_size=128),
                mae=0.01,
                mse=0.001,
            )

            writer.add_tensor(
                name="layer2.weight",
                data=data2,
                scales=scales2,
                original_shape=(128, 256),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=4, group_size=128),
                mae=0.02,
                mse=0.002,
            )

            artifact = writer.write()

            assert (artifact_dir / "manifest.json").exists()
            assert (artifact_dir / "tensors" / "layer1_weight.bin").exists()
            assert (artifact_dir / "tensors" / "layer1_weight.scales.bin").exists()
            assert (artifact_dir / "tensors" / "layer2_weight.bin").exists()
            assert (artifact_dir / "tensors" / "layer2_weight.scales.bin").exists()

            assert artifact.manifest.model.parameter_count == 65536
            assert len(artifact.manifest.tensors) == 2
            assert artifact.manifest.tensors[0].name == "layer1.weight"
            assert artifact.manifest.tensors[1].name == "layer2.weight"

    def test_write_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir1 = Path(tmpdir) / "artifact1"
            artifact_dir2 = Path(tmpdir) / "artifact2"

            for artifact_dir in [artifact_dir1, artifact_dir2]:
                writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
                writer.set_model_info(ModelInfo(
                    model_id="test_model",
                    architecture="test",
                    parameter_count=1000,
                    tensor_count=1,
                ))
                writer.set_compression_info(CompressionInfo(
                    method=CompressionMethod.INT8,
                    target_bits=8,
                    group_size=128,
                ))
                writer.set_compilation_info(CompilationInfo(
                    ldmark_version="0.1.0",
                    format_version="ldmark-artifact-1.0",
                    created_at="2024-01-01T00:00:00",
                ))

                data = np.array([1, 2, 3, 4, 5] * 200, dtype=np.uint8)
                scales = np.array([0.1] * 10, dtype=np.float16)

                writer.add_tensor(
                    name="test.weight",
                    data=data,
                    scales=scales,
                    original_shape=(1000,),
                    original_dtype=np.float16,
                    quantization=QuantizationParams(target_bits=8, group_size=128),
                )

                writer.write()

            with open(artifact_dir1 / "manifest.json") as f:
                manifest1 = json.load(f)
            with open(artifact_dir2 / "manifest.json") as f:
                manifest2 = json.load(f)

            manifest1.pop("compilation", {}).pop("created_at", None)
            manifest2.pop("compilation", {}).pop("created_at", None)
            manifest1.pop("manifest_sha256", None)
            manifest2.pop("manifest_sha256", None)

            assert manifest1 == manifest2

            with open(artifact_dir1 / "tensors" / "test_weight.bin", "rb") as f:
                data1 = f.read()
            with open(artifact_dir2 / "tensors" / "test_weight.bin", "rb") as f:
                data2 = f.read()
            assert data1 == data2


class TestArtifactReader:
    """Tests for LDMARKArtifactReader."""

    @pytest.fixture
    def sample_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="test_model",
                architecture="llama",
                parameter_count=65536,
                tensor_count=2,
                hidden_size=256,
                num_layers=2,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT4,
                target_bits=4,
                group_size=128,
                scale_dtype="float16",
                symmetric=True,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
                optimization_strategy="balanced",
            ))
            writer.set_hardware_info(HardwareInfo(
                cpu_model="Test CPU",
                cpu_cores=8,
                system_ram_gb=16.0,
            ))
            writer.set_validation_info(ValidationInfo(
                status="passed",
                passed=True,
                details={"compression_ratio": 4.0},
            ))

            data1 = np.random.randint(0, 255, size=8192, dtype=np.uint8)
            scales1 = np.random.randn(64).astype(np.float16)

            data2 = np.random.randint(0, 255, size=4096, dtype=np.uint8)
            scales2 = np.random.randn(32).astype(np.float16)

            writer.add_tensor(
                name="layer1.weight",
                data=data1,
                scales=scales1,
                original_shape=(256, 256),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=4, group_size=128),
                mae=0.01,
                mse=0.001,
            )

            writer.add_tensor(
                name="layer2.weight",
                data=data2,
                scales=scales2,
                original_shape=(128, 256),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=4, group_size=128),
                mae=0.02,
                mse=0.002,
            )

            writer.write()
            yield artifact_dir

    def test_open_artifact(self, sample_artifact):
        reader = open_artifact(sample_artifact)
        assert reader.artifact_dir.resolve() == sample_artifact.resolve()

    def test_inspect_artifact(self, sample_artifact):
        info = inspect_artifact(sample_artifact)

        assert info["format_version"] == "ldmark-artifact-1.0"
        assert info["model"]["model_id"] == "test_model"
        assert info["model"]["architecture"] == "llama"
        assert info["model"]["parameter_count"] == 65536
        assert info["tensor_count"] == 2
        assert "layer1.weight" in info["tensor_names"]
        assert "layer2.weight" in info["tensor_names"]
        assert info["compression"]["method"] == "int4"
        assert info["storage"] is not None

    def test_read_tensor(self, sample_artifact):
        reader = open_artifact(sample_artifact)

        tensor = reader.read_tensor("layer1.weight")

        assert tensor.name == "layer1.weight"
        assert tensor.data.size == 8192
        assert tensor.scales is not None
        assert tensor.scales.size == 64
        assert tensor.index_entry.byte_length == 8192
        assert tensor.index_entry.scale_byte_length == 128

    def test_list_tensors(self, sample_artifact):
        reader = open_artifact(sample_artifact)
        names = reader.list_tensors()

        assert len(names) == 2
        assert "layer1.weight" in names
        assert "layer2.weight" in names

    def test_get_tensor_info(self, sample_artifact):
        reader = open_artifact(sample_artifact)
        info = reader.get_tensor_info("layer1.weight")

        assert info is not None
        assert info.name == "layer1.weight"
        assert info.original_shape == [256, 256]
        assert info.quantization.target_bits == 4

    def test_iter_tensors(self, sample_artifact):
        reader = open_artifact(sample_artifact)

        tensors = list(reader.iter_tensors())
        assert len(tensors) == 2

        for name, tensor in tensors:
            assert name in ["layer1.weight", "layer2.weight"]
            assert tensor.data is not None

    def test_read_nonexistent_tensor(self, sample_artifact):
        reader = open_artifact(sample_artifact)

        with pytest.raises(KeyError):
            reader.read_tensor("nonexistent.weight")

    def test_verify_integrity(self, sample_artifact):
        report = verify_artifact(sample_artifact)

        assert report.manifest_valid is True
        assert report.manifest_checksum_valid is True
        assert report.tensor_files_exist is True
        assert report.tensor_checksums_valid is True
        assert report.tensor_sizes_valid is True
        assert report.scale_files_exist is True
        assert report.scale_checksums_valid is True
        assert report.scale_sizes_valid is True
        assert report.all_valid is True
        assert report.total_files_checked == 4  # 2 tensors + 2 scales

    def test_quick_verify(self, sample_artifact):
        assert quick_verify(sample_artifact) is True

        manifest_path = sample_artifact / "manifest.json"
        manifest_path.write_text("{invalid json")
        assert quick_verify(sample_artifact) is False

    def test_missing_tensor_file(self, sample_artifact):
        tensor_path = sample_artifact / "tensors" / "layer1_weight.bin"
        tensor_path.unlink()

        reader = open_artifact(sample_artifact)

        with pytest.raises(MissingFileError):
            reader.read_tensor("layer1.weight")


class TestIntegrityVerification:
    """Tests for artifact integrity verification."""

    @pytest.fixture
    def sample_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="test_model",
                architecture="test",
                parameter_count=1000,
                tensor_count=1,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT8,
                target_bits=8,
                group_size=128,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
            ))

            data = np.array([1, 2, 3, 4, 5] * 200, dtype=np.uint8)
            scales = np.array([0.1] * 10, dtype=np.float16)

            writer.add_tensor(
                name="test.weight",
                data=data,
                scales=scales,
                original_shape=(1000,),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=8, group_size=128),
            )

            writer.write()
            yield artifact_dir

    def test_corrupted_tensor_data(self, sample_artifact):
        tensor_path = sample_artifact / "tensors" / "test_weight.bin"
        tensor_path.write_bytes(b"corrupted data")

        report = verify_artifact(sample_artifact)

        assert report.tensor_checksums_valid is False
        assert report.tensor_sizes_valid is False
        assert len(report.errors) > 0
        assert report.all_valid is False

    def test_corrupted_manifest(self, sample_artifact):
        manifest_path = sample_artifact / "manifest.json"
        with open(manifest_path, "r") as f:
            data = json.load(f)
        data["model"]["model_id"] = "tampered"
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        report = verify_artifact(sample_artifact)

        assert report.manifest_checksum_valid is False
        assert report.all_valid is False

    def test_missing_tensor_file(self, sample_artifact):
        tensor_path = sample_artifact / "tensors" / "test_weight.bin"
        tensor_path.unlink()

        report = verify_artifact(sample_artifact)

        assert report.tensor_files_exist is False
        assert report.all_valid is False

    def test_wrong_format_version(self, sample_artifact):
        manifest_path = sample_artifact / "manifest.json"
        with open(manifest_path, "r") as f:
            data = json.load(f)
        data["format_version"] = "ldmark-artifact-999.0"
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        reader = LDMARKArtifactReader(sample_artifact)

        with pytest.raises(VersionMismatchError):
            _ = reader.manifest


class TestStorageAccounting:
    """Tests for storage accounting."""

    def test_storage_accounting_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="test_model",
                architecture="test",
                parameter_count=1000000,
                tensor_count=2,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT4,
                target_bits=4,
                group_size=128,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
            ))

            data1 = np.random.randint(0, 255, size=500000, dtype=np.uint8)
            scales1 = np.random.randn(4000).astype(np.float16)

            data2 = np.random.randint(0, 255, size=250000, dtype=np.uint8)
            scales2 = np.random.randn(2000).astype(np.float16)

            writer.add_tensor(
                name="layer1.weight",
                data=data1,
                scales=scales1,
                original_shape=(1000, 1000),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=4, group_size=128),
            )
            writer.add_tensor(
                name="layer2.weight",
                data=data2,
                scales=scales2,
                original_shape=(500, 1000),
                original_dtype=np.float16,
                quantization=QuantizationParams(target_bits=4, group_size=128),
            )

            artifact = writer.write()

            storage = artifact.manifest.storage
            assert storage is not None
            assert storage.raw_tensor_bytes == 750000
            assert storage.scale_bytes == 12000
            assert storage.total_artifact_bytes > storage.raw_tensor_bytes + storage.scale_bytes
            assert storage.actual_bits_per_weight > 0
            assert storage.theoretical_bits_per_weight > 0


class TestRoundTrip:
    """Tests for round-trip artifact creation and reading."""

    def test_round_trip_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="round_trip_model",
                architecture="mistral",
                parameter_count=5000000,
                tensor_count=20,
                num_layers=24,
                hidden_size=1024,
                vocab_size=32000,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT8,
                target_bits=8,
                group_size=128,
                symmetric=False,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
                optimization_strategy="accuracy",
            ))
            writer.set_hardware_info(HardwareInfo(
                cpu_model="Intel i9",
                cpu_cores=16,
                system_ram_gb=64.0,
                gpu_model="NVIDIA RTX 4090",
                gpu_vram_gb=24.0,
                gpu_vendor="nvidia",
            ))
            writer.set_validation_info(ValidationInfo(
                status="passed",
                passed=True,
            ))

            for i in range(3):
                data = np.random.randint(0, 255, size=10000, dtype=np.uint8)
                scales = np.random.randn(80).astype(np.float16)
                writer.add_tensor(
                    name=f"layer{i}.weight",
                    data=data,
                    scales=scales,
                    original_shape=(100, 100),
                    original_dtype=np.float16,
                    quantization=QuantizationParams(target_bits=8, group_size=128),
                )

            writer.write()

            reader = open_artifact(artifact_dir)

            model = reader.get_model_info()
            assert model.model_id == "round_trip_model"
            assert model.architecture == "mistral"
            assert model.parameter_count == 5000000
            assert model.num_layers == 24
            assert model.hidden_size == 1024
            assert model.vocab_size == 32000

            comp = reader.get_compression_info()
            assert comp.method == CompressionMethod.INT8
            assert comp.target_bits == 8
            assert comp.group_size == 128
            assert comp.symmetric is False

            hw = reader.get_hardware_info()
            assert hw.cpu_model == "Intel i9"
            assert hw.cpu_cores == 16
            assert hw.gpu_model == "NVIDIA RTX 4090"

            comp_info = reader.get_compilation_info()
            assert comp_info.optimization_strategy == "accuracy"

            val = reader.get_validation_info()
            assert val.status == "passed"
            assert val.passed is True

            assert reader.get_tensor_count() == 3
            names = reader.list_tensors()
            assert "layer0.weight" in names
            assert "layer1.weight" in names
            assert "layer2.weight" in names


class TestEmptyMinimalArtifacts:
    """Tests for edge cases like empty/minimal artifacts."""

    def test_empty_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "empty_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="empty_model",
                architecture="test",
                parameter_count=0,
                tensor_count=0,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.FP16,
                target_bits=16,
                group_size=0,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
            ))

            artifact = writer.write()

            assert artifact.manifest.model.parameter_count == 0
            assert len(artifact.manifest.tensors) == 0
            assert artifact.manifest.storage.raw_tensor_bytes == 0
            assert artifact.manifest.storage.scale_bytes == 0

    def test_single_tensor_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "single_tensor"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="single_tensor_model",
                architecture="test",
                parameter_count=100,
                tensor_count=1,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT8,
                target_bits=8,
                group_size=128,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
            ))

            data = np.array([1, 2, 3, 4, 5] * 20, dtype=np.uint8)
            writer.add_tensor(
                name="only.weight",
                data=data,
                original_shape=(100,),
                original_dtype=np.float16,
            )

            artifact = writer.write()

            assert len(artifact.manifest.tensors) == 1
            assert artifact.manifest.tensors[0].name == "only.weight"

            reader = open_artifact(artifact_dir)
            tensor = reader.read_tensor("only.weight")
            assert tensor.data.size == 100


class TestVersionCompatibility:
    """Tests for format version compatibility."""

    def test_format_version_enum(self):
        assert ArtifactFormatVersion.V1.value == "ldmark-artifact-1.0"
        assert ArtifactFormatVersion.latest() == ArtifactFormatVersion.V1
        assert ArtifactFormatVersion.from_string("ldmark-artifact-1.0") == ArtifactFormatVersion.V1

        with pytest.raises(ValueError):
            ArtifactFormatVersion.from_string("ldmark-artifact-2.0")

    def test_unsupported_version_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "test_artifact"

            writer = LDMARKArtifactWriter(artifact_dir, overwrite=True)
            writer.set_model_info(ModelInfo(
                model_id="test",
                architecture="test",
                parameter_count=100,
                tensor_count=1,
            ))
            writer.set_compression_info(CompressionInfo(
                method=CompressionMethod.INT8,
                target_bits=8,
                group_size=128,
            ))
            writer.set_compilation_info(CompilationInfo(
                ldmark_version="0.1.0",
                format_version="ldmark-artifact-1.0",
                created_at="2024-01-01T00:00:00",
            ))

            data = np.array([1, 2, 3, 4, 5] * 20, dtype=np.uint8)
            writer.add_tensor(
                name="test.weight",
                data=data,
                original_shape=(100,),
                original_dtype=np.float16,
            )

            writer.write()

            manifest_path = artifact_dir / "manifest.json"
            with open(manifest_path, "r") as f:
                manifest_data = json.load(f)
            manifest_data["format_version"] = "ldmark-artifact-999.0"
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f)

            reader = LDMARKArtifactReader(artifact_dir)
            with pytest.raises(VersionMismatchError):
                _ = reader.manifest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])