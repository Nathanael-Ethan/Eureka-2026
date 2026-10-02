"""
Integration tests for LDMARK Real Model End-to-End Pipeline.

Tests the full pipeline on a real model fixture:
- real model loading
- analysis
- INT8 compilation
- INT4 compilation
- validation
- export
- result serialization
- deterministic repeated runs
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from ldmark.compiler.config import (
    CompilerConfig,
    CompressionMethod,
    OptimizationStrategy,
    PipelineStage,
    PrecisionTarget,
    TargetHardware,
    ValidationConfig,
)
from ldmark.compiler.pipeline import Pipeline, StageStatus
from ldmark.compiler.stages import register_default_stages
from ldmark.compiler.result import CompilationArtifact, ValidationStatus
from ldmark.file_io.factory import load_model


# Test fixture paths
FIXTURE_DIR = Path(__file__).parent.parent.parent / "experiments" / "real_model"
MODEL_PATH = FIXTURE_DIR / "tiny_llama_2L_128H.npz"
CONFIG_PATH = FIXTURE_DIR / "tiny_llama_2L_128H.json"


@pytest.fixture(scope="module")
def model_fixture_exists():
    """Check if model fixture exists."""
    return MODEL_PATH.exists() and CONFIG_PATH.exists()


@pytest.fixture(scope="module")
def model_fixture():
    """Load model fixture for testing."""
    if not MODEL_PATH.exists():
        pytest.skip("Model fixture not found. Run create_fixture.py first.")
    return load_model(MODEL_PATH)


class TestRealModelLoading:
    """Test real model loading from fixture."""
    
    def test_model_file_exists(self, model_fixture_exists):
        assert model_fixture_exists, f"Model fixture not found at {MODEL_PATH}"
    
    def test_load_model(self, model_fixture):
        assert model_fixture is not None
        assert model_fixture.metadata is not None
        assert model_fixture.metadata.total_parameters > 0
        assert model_fixture.metadata.tensor_count > 0
        state_dict = model_fixture.to_state_dict()
        assert len(state_dict) > 0
    
    def test_model_parameters(self, model_fixture):
        # Tiny Llama: ~787K parameters
        assert model_fixture.metadata.total_parameters == 787072
    
    def test_model_tensors(self, model_fixture):
        assert model_fixture.metadata.tensor_count == 17
        # Check key tensors exist
        state_dict = model_fixture.to_state_dict()
        tensor_names = set(state_dict.keys())
        assert "model.embed_tokens.weight" in tensor_names
        assert "lm_head.weight" in tensor_names
        assert "model.layers.0.self_attn.qkv_proj.weight" in tensor_names
        assert "model.layers.1.mlp.down_proj.weight" in tensor_names
    
    def test_model_dtype(self, model_fixture):
        # All tensors should be float16
        state_dict = model_fixture.to_state_dict()
        for name, tensor in state_dict.items():
            assert tensor.dtype == np.float16, f"Tensor {name} has dtype {tensor.dtype}"
    
    def test_config_sidecar(self):
        assert CONFIG_PATH.exists()
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        assert config["model_type"] == "llama"
        assert config["num_hidden_layers"] == 2
        assert config["hidden_size"] == 128
        assert config["vocab_size"] == 1024


class TestRealModelAnalysis:
    """Test model analysis on real model."""
    
    def test_analyze_stage(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / "out",
                compression_method=CompressionMethod.INT8,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            
            # Run just load and analyze
            load_stage = pipeline.get_stage(PipelineStage.LOAD)
            analyze_stage = pipeline.get_stage(PipelineStage.ANALYZE)
            
            from ldmark.compiler.pipeline import PipelineContext
            context = PipelineContext(config=config)
            
            loaded = load_stage.run(str(MODEL_PATH), context)
            result = analyze_stage.run(loaded, context)
            
            analysis = result["analysis"]
            assert analysis.parameter_counts.total == 787072
            assert len(analysis.tensors) == 17
            assert analysis.architecture is not None
    
    def test_analysis_with_config(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / "out",
                compression_method=CompressionMethod.INT8,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            
            from ldmark.compiler.pipeline import PipelineContext
            context = PipelineContext(config=config)
            
            load_stage = pipeline.get_stage(PipelineStage.LOAD)
            analyze_stage = pipeline.get_stage(PipelineStage.ANALYZE)
            
            loaded = load_stage.run(str(MODEL_PATH), context)
            
            # Inject config
            with open(CONFIG_PATH) as f:
                config_dict = json.load(f)
            loaded["config"] = config_dict
            context.set_artifact("loaded_model", loaded)
            
            result = analyze_stage.run(loaded, context)
            analysis = result["analysis"]
            
            # Architecture info should be populated from config
            assert analysis.architecture.num_layers == 2
            assert analysis.architecture.hidden_size == 128
            assert analysis.architecture.vocab_size == 1024


class TestRealModelINT8Compilation:
    """Test INT8 compilation on real model."""
    
    def test_int8_full_pipeline(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_int8"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT8,
                optimization_strategy=OptimizationStrategy.BALANCED,
                validation=ValidationConfig(enabled=True),
                verbose=False,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            # Check all stages completed
            for stage in PipelineStage:
                result = context.get_stage_result(stage)
                assert result.status == StageStatus.COMPLETED, f"Stage {stage.value} failed: {result.error}"
            
            # Check artifact
            artifact = context.get_artifact("artifact")
            assert artifact is not None
            assert artifact.compression_method == "int8"
            assert artifact.target_bits == 8
            assert artifact.parameter_count == 787072
            assert len(artifact.tensors) == 17
            assert artifact.overall_compression_ratio > 1.5  # INT8 should compress
            
            # Check validation passed
            validation = context.get_artifact("validation")
            assert validation.status == ValidationStatus.PASSED
            assert validation.details["storage_reduced"] is True
            assert validation.details["shape_preserved"] is True
    
    def test_int8_error_metrics(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_int8"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT8,
                validation=ValidationConfig(enabled=True),
                verbose=False,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            artifact = context.get_artifact("artifact")
            
            # Check per-tensor metrics
            for tensor_info in artifact.tensors:
                assert tensor_info.mae >= 0.0
                assert tensor_info.mse >= 0.0
                assert tensor_info.max_abs_error >= 0.0
                assert tensor_info.relative_error >= 0.0
                assert tensor_info.bits_per_weight > 8.0  # Should include scale overhead
                assert tensor_info.target_bits == 8
                assert tensor_info.group_size == 128


class TestRealModelINT4Compilation:
    """Test INT4 compilation on real model."""
    
    def test_int4_full_pipeline(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_int4"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT4,
                optimization_strategy=OptimizationStrategy.BALANCED,
                validation=ValidationConfig(enabled=True),
                verbose=False,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            # Check all stages completed
            for stage in PipelineStage:
                result = context.get_stage_result(stage)
                assert result.status == StageStatus.COMPLETED, f"Stage {stage.value} failed: {result.error}"
            
            artifact = context.get_artifact("artifact")
            assert artifact is not None
            assert artifact.compression_method == "int4"
            assert artifact.target_bits == 4
            assert artifact.overall_compression_ratio > 3.0  # INT4 should compress more
            
            validation = context.get_artifact("validation")
            assert validation.status == ValidationStatus.PASSED
    
    def test_int4_bits_per_weight(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_int4"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT4,
                validation=ValidationConfig(enabled=True),
                verbose=False,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            artifact = context.get_artifact("artifact")
            
            # INT4 with group_size=128 and fp16 scales should be ~4.125 bits/weight
            assert 4.0 < artifact.overall_bits_per_weight < 4.5


class TestRealModelValidation:
    """Test validation stage on real model."""
    
    def test_validation_passes_for_int8(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / "out",
                compression_method=CompressionMethod.INT8,
                validation=ValidationConfig(enabled=True),
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            validation = context.get_artifact("validation")
            assert validation.status == ValidationStatus.PASSED
            assert validation.details["count_preserved"] is True
            assert validation.details["storage_reduced"] is True
    
    def test_validation_skipped_when_disabled(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / "out",
                compression_method=CompressionMethod.INT8,
                validation=ValidationConfig(enabled=False),
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            validation = context.get_artifact("validation")
            assert validation.status == ValidationStatus.SKIPPED


class TestRealModelExport:
    """Test export stage on real model."""
    
    def test_export_creates_files(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT8,
                validation=ValidationConfig(enabled=True),
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            export_result = context.get_stage_result(PipelineStage.EXPORT)
            assert export_result.status == StageStatus.COMPLETED
            
            # Check files exist
            metadata_path = output_path / "compilation_metadata.json"
            tensors_dir = output_path / "tensors"
            
            assert metadata_path.exists()
            assert tensors_dir.exists()
            assert len(list(tensors_dir.glob("*.npz"))) == 17
    
    def test_export_metadata_valid_json(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT8,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            metadata_path = output_path / "compilation_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert metadata["format_version"] == "ldmark-compilation-0.1"
            assert metadata["is_experimental"] is True
            assert metadata["compression_method"] == "int8"
            assert metadata["parameter_count"] == 787072
            assert len(metadata["tensors"]) == 17
            assert metadata["overall_compression_ratio"] > 1.0
    
    def test_export_tensor_files_loadable(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled"
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=output_path,
                compression_method=CompressionMethod.INT8,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            tensors_dir = output_path / "tensors"
            for npz_file in tensors_dir.glob("*.npz"):
                data = np.load(npz_file)
                assert "data" in data
                assert "scales" in data
                assert "original_shape" in data
                # Verify data can be loaded
                compressed_data = data["data"]
                scales = data["scales"]
                assert compressed_data.size > 0
                assert scales.size > 0


class TestRealModelResultSerialization:
    """Test result serialization for real model."""
    
    def test_artifact_to_json(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / "out",
                compression_method=CompressionMethod.INT8,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            artifact = context.get_artifact("artifact")
            json_str = artifact.to_json()
            
            # Should be valid JSON
            parsed = json.loads(json_str)
            assert parsed["format_version"] == "ldmark-compilation-0.1"
            assert parsed["compression_method"] == "int8"
            assert len(parsed["tensors"]) == 17
    
    def test_artifact_roundtrip(self, model_fixture):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / "out",
                compression_method=CompressionMethod.INT8,
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            artifact = context.get_artifact("artifact")
            json_str = artifact.to_json()
            restored = CompilationArtifact.from_dict(json.loads(json_str))
            
            assert restored.model_id == artifact.model_id
            assert restored.compression_method == artifact.compression_method
            assert restored.target_bits == artifact.target_bits
            assert restored.parameter_count == artifact.parameter_count
            assert len(restored.tensors) == len(artifact.tensors)
            assert restored.overall_compression_ratio == artifact.overall_compression_ratio


class TestRealModelDeterministicRuns:
    """Test deterministic results with fixed seed."""
    
    def test_int8_deterministic(self, model_fixture):
        """Two runs with same seed should produce identical compressed tensors."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            config1 = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir1) / "out",
                compression_method=CompressionMethod.INT8,
                validation=ValidationConfig(enabled=True),
            )
            config2 = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir2) / "out",
                compression_method=CompressionMethod.INT8,
                validation=ValidationConfig(enabled=True),
            )
            
            pipeline1 = Pipeline(config1)
            register_default_stages(pipeline1, config1)
            context1 = pipeline1.run(initial_input=str(MODEL_PATH))
            
            pipeline2 = Pipeline(config2)
            register_default_stages(pipeline2, config2)
            context2 = pipeline2.run(initial_input=str(MODEL_PATH))
            
            artifact1 = context1.get_artifact("artifact")
            artifact2 = context2.get_artifact("artifact")
            
            # Compression ratios should be identical
            assert artifact1.overall_compression_ratio == artifact2.overall_compression_ratio
            assert artifact1.overall_bits_per_weight == artifact2.overall_bits_per_weight
            
            # Per-tensor metrics should be identical
            for t1, t2 in zip(artifact1.tensors, artifact2.tensors):
                assert t1.mae == t2.mae
                assert t1.mse == t2.mse
                assert t1.max_abs_error == t2.max_abs_error
                assert t1.relative_error == t2.relative_error
                assert t1.bits_per_weight == t2.bits_per_weight
    
    def test_int4_deterministic(self, model_fixture):
        """Two runs with same seed should produce identical compressed tensors."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            config1 = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir1) / "out",
                compression_method=CompressionMethod.INT4,
                validation=ValidationConfig(enabled=True),
            )
            config2 = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir2) / "out",
                compression_method=CompressionMethod.INT4,
                validation=ValidationConfig(enabled=True),
            )
            
            pipeline1 = Pipeline(config1)
            register_default_stages(pipeline1, config1)
            context1 = pipeline1.run(initial_input=str(MODEL_PATH))
            
            pipeline2 = Pipeline(config2)
            register_default_stages(pipeline2, config2)
            context2 = pipeline2.run(initial_input=str(MODEL_PATH))
            
            artifact1 = context1.get_artifact("artifact")
            artifact2 = context2.get_artifact("artifact")
            
            assert artifact1.overall_compression_ratio == artifact2.overall_compression_ratio
            assert artifact1.overall_bits_per_weight == artifact2.overall_bits_per_weight
            
            for t1, t2 in zip(artifact1.tensors, artifact2.tensors):
                assert t1.mae == t2.mae
                assert t1.mse == t2.mse


class TestRealModelMultiplePrecisions:
    """Test multiple compression precisions on real model."""
    
    @pytest.mark.parametrize("method,expected_ratio_min", [
        (CompressionMethod.INT8, 1.5),
        (CompressionMethod.INT4, 3.0),
        (CompressionMethod.BINARY, 10.0),
        (CompressionMethod.TERNARY, 5.0),
    ])
    def test_compression_ratios(self, model_fixture, method, expected_ratio_min):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompilerConfig(
                input_model_path=MODEL_PATH,
                output_path=Path(tmpdir) / f"out_{method.value}",
                compression_method=method,
                validation=ValidationConfig(enabled=True),
            )
            
            pipeline = Pipeline(config)
            register_default_stages(pipeline, config)
            context = pipeline.run(initial_input=str(MODEL_PATH))
            
            assert context.get_stage_result(PipelineStage.TRANSFORM).status == StageStatus.COMPLETED
            
            artifact = context.get_artifact("artifact")
            assert artifact.overall_compression_ratio >= expected_ratio_min
            assert artifact.compression_method == method.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])