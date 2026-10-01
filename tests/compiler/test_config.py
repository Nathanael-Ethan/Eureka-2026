"""
Tests for LDMARK Compiler Configuration
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.ldmark.compiler.config import (
    CompilerConfig,
    HardwareProfile,
    OptimizationStrategy,
    PipelineStage,
    PrecisionTarget,
    TargetHardware,
    ValidationConfig,
)


class TestHardwareProfile:
    def test_default_profile(self):
        profile = HardwareProfile(name="test_device")
        assert profile.name == "test_device"
        assert profile.cpu_cores is None
        assert profile.supported_precisions == []

    def test_full_profile(self):
        profile = HardwareProfile(
            name="laptop",
            cpu_cores=8,
            cpu_arch="x86_64",
            gpu_memory_gb=8.0,
            system_memory_gb=16.0,
            supported_precisions=[PrecisionTarget.FP16, PrecisionTarget.INT8],
        )
        assert profile.cpu_cores == 8
        assert len(profile.supported_precisions) == 2

    def test_serialization(self):
        profile = HardwareProfile(
            name="test",
            cpu_cores=4,
            supported_precisions=[PrecisionTarget.FP16],
        )
        data = profile.to_dict()
        assert data["name"] == "test"
        assert data["cpu_cores"] == 4
        assert data["supported_precisions"] == ["fp16"]

        restored = HardwareProfile.from_dict(data)
        assert restored.name == "test"
        assert restored.cpu_cores == 4
        assert restored.supported_precisions == [PrecisionTarget.FP16]


class TestValidationConfig:
    def test_default_config(self):
        config = ValidationConfig()
        assert config.enabled is True
        assert config.tolerance_rtol == 1e-3
        assert config.tolerance_atol == 1e-5

    def test_custom_config(self):
        config = ValidationConfig(
            enabled=False,
            tolerance_rtol=1e-2,
            max_tokens=50,
        )
        assert config.enabled is False
        assert config.tolerance_rtol == 1e-2
        assert config.max_tokens == 50


class TestCompilerConfig:
    def test_minimal_config(self):
        config = CompilerConfig(
            input_model_path=Path("/models/test"),
            output_path=Path("/output/test"),
        )
        assert config.input_model_path == Path("/models/test")
        assert config.output_path == Path("/output/test")
        assert config.target_hardware == TargetHardware.UNKNOWN
        assert config.target_precision == PrecisionTarget.AUTO
        assert config.optimization_strategy == OptimizationStrategy.BALANCED

    def test_full_config(self):
        config = CompilerConfig(
            input_model_path=Path("/models/llama"),
            output_path=Path("/output/llama_compressed"),
            target_hardware=TargetHardware.LAPTOP_CPU,
            max_model_size_gb=4.0,
            target_precision=PrecisionTarget.INT4,
            optimization_strategy=OptimizationStrategy.SIZE,
            verbose=True,
            dry_run=True,
        )
        assert config.max_model_size_gb == 4.0
        assert config.target_precision == PrecisionTarget.INT4
        assert config.optimization_strategy == OptimizationStrategy.SIZE
        assert config.verbose is True
        assert config.dry_run is True

    def test_stage_control(self):
        config = CompilerConfig(
            input_model_path=Path("/models/test"),
            output_path=Path("/output/test"),
            enabled_stages=[PipelineStage.LOAD, PipelineStage.ANALYZE],
            skip_stages=[PipelineStage.VALIDATE],
        )
        assert config.is_stage_enabled(PipelineStage.LOAD) is True
        assert config.is_stage_enabled(PipelineStage.ANALYZE) is True
        assert config.is_stage_enabled(PipelineStage.VALIDATE) is False
        assert config.is_stage_enabled(PipelineStage.TRANSFORM) is False

    def test_stage_config(self):
        config = CompilerConfig(
            input_model_path=Path("/models/test"),
            output_path=Path("/output/test"),
            stage_config={
                "analyze": {"detail_level": "full"},
                "transform": {"method": "quantize"},
            },
        )
        analyze_cfg = config.get_stage_config(PipelineStage.ANALYZE)
        assert analyze_cfg == {"detail_level": "full"}

    def test_serialization(self):
        config = CompilerConfig(
            input_model_path=Path("/models/test"),
            output_path=Path("/output/test"),
            target_hardware=TargetHardware.LAPTOP_GPU,
            max_model_size_gb=8.0,
        )
        data = config.to_dict()
        assert Path(data["input_model_path"]).as_posix() == "/models/test"
        assert data["target_hardware"] == "laptop_gpu"
        assert data["max_model_size_gb"] == 8.0

        restored = CompilerConfig.from_dict(data)
        assert restored.input_model_path.as_posix() == "/models/test"
        assert restored.target_hardware == TargetHardware.LAPTOP_GPU
        assert restored.max_model_size_gb == 8.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])