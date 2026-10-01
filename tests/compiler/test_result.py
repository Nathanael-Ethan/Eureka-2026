"""
Tests for LDMARK Compiler Result Structures
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pytest

from src.ldmark.compiler.result import (
    CompilationResult,
    CompilationStatus,
    ModelReference,
    RuntimeMemoryMetrics,
    StorageMetrics,
    ValidationResult,
    ValidationStatus,
)


class TestModelReference:
    def test_minimal_reference(self):
        ref = ModelReference(path=Path("/models/test"))
        assert ref.path == Path("/models/test")
        assert ref.size_bytes is None

    def test_full_reference(self):
        ref = ModelReference(
            path=Path("/models/llama"),
            format="gguf",
            size_bytes=4_000_000_000,
            parameter_count=7_000_000_000,
            architecture="llama",
        )
        assert ref.format == "gguf"
        assert ref.size_bytes == 4_000_000_000

    def test_serialization(self):
        ref = ModelReference(
            path=Path("/models/test"),
            format="safetensors",
            size_bytes=1000,
        )
        data = ref.to_dict()
        assert Path(data["path"]).as_posix() == "/models/test"
        assert data["format"] == "safetensors"
        assert data["size_bytes"] == 1000


class TestStorageMetrics:
    def test_default_metrics(self):
        metrics = StorageMetrics()
        assert metrics.original_size_bytes is None
        assert metrics.compression_ratio is None

    def test_with_values(self):
        metrics = StorageMetrics(
            original_size_bytes=8_000_000_000,
            compressed_size_bytes=2_000_000_000,
            compression_ratio=4.0,
            bits_per_weight=4.0,
        )
        assert metrics.compression_ratio == 4.0
        assert metrics.bits_per_weight == 4.0

    def test_serialization(self):
        metrics = StorageMetrics(
            original_size_bytes=1000,
            compressed_size_bytes=250,
            compression_ratio=4.0,
        )
        data = metrics.to_dict()
        assert data["compression_ratio"] == 4.0


class TestRuntimeMemoryMetrics:
    def test_default_metrics(self):
        metrics = RuntimeMemoryMetrics()
        assert metrics.total_gb is None

    def test_with_values(self):
        metrics = RuntimeMemoryMetrics(
            weights_gb=2.0,
            kv_cache_gb=1.0,
            activations_gb=0.5,
            overhead_gb=0.3,
            total_gb=3.8,
        )
        assert metrics.total_gb == 3.8


class TestValidationResult:
    def test_default_result(self):
        result = ValidationResult()
        assert result.status == ValidationStatus.NOT_RUN
        assert result.test_cases_passed == 0

    def test_passed_result(self):
        result = ValidationResult(
            status=ValidationStatus.PASSED,
            logits_match=True,
            test_cases_passed=10,
            test_cases_total=10,
        )
        assert result.status == ValidationStatus.PASSED
        assert result.logits_match is True

    def test_serialization(self):
        result = ValidationResult(
            status=ValidationStatus.FAILED,
            error="Mismatch in layer 5",
        )
        data = result.to_dict()
        assert data["status"] == "failed"
        assert data["error"] == "Mismatch in layer 5"


class TestCompilationResult:
    def test_default_result(self):
        result = CompilationResult()
        assert result.status == CompilationStatus.FAILED
        assert result.started_at is not None
        assert result.completed_at is None

    def test_mark_completed(self):
        result = CompilationResult()
        result.mark_completed(CompilationStatus.SUCCESS)

        assert result.status == CompilationStatus.SUCCESS
        assert result.completed_at is not None
        assert result.duration_seconds is not None
        assert result.duration_seconds >= 0

    def test_add_error(self):
        result = CompilationResult()
        result.add_error("Test error")

        assert result.status == CompilationStatus.FAILED
        assert "Test error" in result.errors

    def test_add_warning(self):
        result = CompilationResult()
        result.add_warning("Test warning")

        assert "Test warning" in result.warnings

    def test_full_result(self):
        result = CompilationResult(
            status=CompilationStatus.SUCCESS,
            input_model=ModelReference(path=Path("/models/input")),
            output_model=ModelReference(path=Path("/models/output")),
            storage=StorageMetrics(
                original_size_bytes=8_000_000_000,
                compressed_size_bytes=2_000_000_000,
                compression_ratio=4.0,
            ),
            runtime_memory=RuntimeMemoryMetrics(total_gb=4.0),
            selected_strategy="size",
        )
        result.mark_completed()

        data = result.to_dict()
        assert data["status"] == "success"
        assert Path(data["input_model"]["path"]).as_posix() == "/models/input"
        assert data["storage"]["compression_ratio"] == 4.0
        assert data["runtime_memory"]["total_gb"] == 4.0
        assert data["selected_strategy"] == "size"

    def test_json_serialization(self):
        result = CompilationResult(
            status=CompilationStatus.SUCCESS,
            input_model=ModelReference(path=Path("/models/test")),
        )
        result.mark_completed()

        json_str = result.to_json()
        assert "success" in json_str
        # Path gets serialized with escaped backslashes in JSON
        assert "models" in json_str and "test" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])