"""
Integration tests for LDMARK Compiler Pipeline

Tests the full pipeline: LOAD -> ANALYZE -> PLAN -> TRANSFORM -> VALIDATE -> EXPORT
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pytest

from src.ldmark.compiler.config import (
    CompilerConfig,
    CompressionMethod,
    OptimizationStrategy,
    PipelineStage,
    PrecisionTarget,
    TargetHardware,
    ValidationConfig,
)
from src.ldmark.compiler.pipeline import Pipeline, StageStatus
from src.ldmark.compiler.result import (
    CompilationArtifact,
    CompilationStatus,
    ValidationStatus,
)
from src.ldmark.compiler.stages import (
    AnalyzeStage,
    ExportStage,
    LoadStage,
    PlanStage,
    TransformStage,
    ValidateStage,
    register_default_stages,
)


def create_synthetic_npz(path: Path, num_tensors: int = 3, size: int = 1000) -> Path:
    """Create a synthetic numpy model file for testing."""
    tensors = {}
    for i in range(num_tensors):
        tensors[f"layer.{i}.weight"] = np.random.randn(size).astype(np.float32)
    np.savez(path, **tensors)
    return path


class TestLoadStage:
    def test_load_numpy_npz(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(input_model_path=model_path, output_path=tmp_path / "out")
        stage = LoadStage(PipelineStage.LOAD)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        result = stage.run(str(model_path), context)

        assert result["loaded"] is True
        assert result["format"] == "numpy"
        assert len(result["tensors"]) == 3

    def test_load_nonexistent(self, tmp_path):
        config = CompilerConfig(input_model_path=tmp_path / "nonexistent.npz", output_path=tmp_path / "out")
        stage = LoadStage(PipelineStage.LOAD)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        with pytest.raises(FileNotFoundError):
            stage.run(str(tmp_path / "nonexistent.npz"), context)

    def test_load_unsupported_format(self, tmp_path):
        model_path = tmp_path / "model.xyz"
        model_path.write_text("dummy")
        config = CompilerConfig(input_model_path=model_path, output_path=tmp_path / "out")
        stage = LoadStage(PipelineStage.LOAD)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        with pytest.raises(ValueError, match="Unsupported model format"):
            stage.run(str(model_path), context)


class TestAnalyzeStage:
    def test_analyze_synthetic_model(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(input_model_path=model_path, output_path=tmp_path / "out")

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        result = analyze_stage.run(loaded, context)

        assert "analysis" in result
        analysis = result["analysis"]
        assert analysis.parameter_counts.total > 0
        assert len(analysis.tensors) == 3
        assert analysis.architecture is not None

    def test_analyze_no_tensors(self, tmp_path):
        config = CompilerConfig(input_model_path=tmp_path / "model.npz", output_path=tmp_path / "out")
        stage = AnalyzeStage(PipelineStage.ANALYZE)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        with pytest.raises(ValueError, match="No tensors available"):
            stage.run({"tensors": {}}, context)


class TestPlanStage:
    def test_plan_int8(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT8,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        result = plan_stage.run(analyzed, context)

        plan = result["plan"]
        assert plan.method == CompressionMethod.INT8
        assert plan.target_bits == 8
        assert plan.group_size == 128
        assert plan.expected_bits_per_weight > 8.0
        assert plan.expected_storage_bytes > 0
        assert plan.expected_compression_ratio > 1.0

    def test_plan_int4(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT4,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        result = plan_stage.run(analyzed, context)

        plan = result["plan"]
        assert plan.method == CompressionMethod.INT4
        assert plan.target_bits == 4

    def test_plan_binary(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.BINARY,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        result = plan_stage.run(analyzed, context)

        plan = result["plan"]
        assert plan.method == CompressionMethod.BINARY
        assert plan.target_bits == 1
        assert len(plan.warnings) > 0

    def test_plan_ternary(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.TERNARY,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        result = plan_stage.run(analyzed, context)

        plan = result["plan"]
        assert plan.method == CompressionMethod.TERNARY
        assert plan.target_bits == 2

    def test_plan_auto_selects_int8(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.AUTO,
            optimization_strategy=OptimizationStrategy.BALANCED,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        result = plan_stage.run(analyzed, context)

        plan = result["plan"]
        assert plan.method == CompressionMethod.INT8


class TestTransformStage:
    def test_transform_int8(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT8,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        result = transform_stage.run(planned, context)

        transform_result = result["transform_result"]
        assert transform_result["num_tensors_compressed"] == 3
        assert transform_result["total_original_size_bytes"] > 0
        assert transform_result["total_compressed_size_bytes"] > 0
        assert len(transform_result["tensor_infos"]) == 3

        for info in transform_result["tensor_infos"]:
            assert info.compression_method == "int8"
            assert info.target_bits == 8
            assert info.mae >= 0.0
            assert info.relative_error >= 0.0

    def test_transform_int4(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT4,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        result = transform_stage.run(planned, context)

        transform_result = result["transform_result"]
        assert transform_result["num_tensors_compressed"] == 3

        for info in transform_result["tensor_infos"]:
            assert info.compression_method == "int4"
            assert info.target_bits == 4

    def test_transform_binary(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.BINARY,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        result = transform_stage.run(planned, context)

        transform_result = result["transform_result"]
        assert transform_result["num_tensors_compressed"] == 3

        for info in transform_result["tensor_infos"]:
            assert info.compression_method == "binary"
            assert info.target_bits == 1

    def test_transform_ternary(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.TERNARY,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        result = transform_stage.run(planned, context)

        transform_result = result["transform_result"]
        assert transform_result["num_tensors_compressed"] == 3

        for info in transform_result["tensor_infos"]:
            assert info.compression_method == "ternary"
            assert info.target_bits == 2


class TestValidateStage:
    def test_validate_passed(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT8,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        validate_stage = ValidateStage(PipelineStage.VALIDATE)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        transformed = transform_stage.run(planned, context)
        result = validate_stage.run(transformed, context)

        validation = result["validation"]
        assert validation.status == ValidationStatus.PASSED
        assert validation.details["shape_preserved"] is True
        assert validation.details["count_preserved"] is True
        assert validation.details["storage_reduced"] is True

    def test_validate_skipped(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT8,
            validation=ValidationConfig(enabled=False),
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        validate_stage = ValidateStage(PipelineStage.VALIDATE)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        transformed = transform_stage.run(planned, context)
        result = validate_stage.run(transformed, context)

        validation = result["validation"]
        assert validation.status == ValidationStatus.SKIPPED


class TestExportStage:
    def test_export_creates_artifact(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.INT8,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        validate_stage = ValidateStage(PipelineStage.VALIDATE)
        export_stage = ExportStage(PipelineStage.EXPORT)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        transformed = transform_stage.run(planned, context)
        validated = validate_stage.run(transformed, context)
        result = export_stage.run(validated, context)

        export_result = result["export_result"]
        assert export_result["exported"] is True
        assert export_result["format"] == "ldmark-compilation-0.1 (experimental)"
        assert Path(export_result["metadata_path"]).exists()
        assert Path(export_result["tensors_dir"]).exists()

        artifact = context.get_artifact("artifact")
        assert artifact is not None
        assert artifact.format_version == "ldmark-compilation-0.1"
        assert artifact.is_experimental is True
        assert artifact.compression_method == "int8"
        assert len(artifact.tensors) == 3
        assert artifact.total_original_size_bytes > 0
        assert artifact.total_compressed_size_bytes > 0
        assert artifact.overall_compression_ratio > 1.0

    def test_export_creates_metadata_json(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.INT8,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        validate_stage = ValidateStage(PipelineStage.VALIDATE)
        export_stage = ExportStage(PipelineStage.EXPORT)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)
        planned = plan_stage.run(analyzed, context)
        transformed = transform_stage.run(planned, context)
        validated = validate_stage.run(transformed, context)
        export_stage.run(validated, context)

        metadata_path = output_path / "compilation_metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["format_version"] == "ldmark-compilation-0.1"
        assert metadata["is_experimental"] is True
        assert metadata["compression_method"] == "int8"
        assert len(metadata["tensors"]) == 3


class TestFullPipeline:
    def test_full_pipeline_int8(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.INT8,
        )

        pipeline = Pipeline(config)
        register_default_stages(pipeline, config)

        context = pipeline.run(initial_input=str(model_path))

        load_result = context.get_stage_result(PipelineStage.LOAD)
        analyze_result = context.get_stage_result(PipelineStage.ANALYZE)
        plan_result = context.get_stage_result(PipelineStage.PLAN)
        transform_result = context.get_stage_result(PipelineStage.TRANSFORM)
        validate_result = context.get_stage_result(PipelineStage.VALIDATE)
        export_result = context.get_stage_result(PipelineStage.EXPORT)

        assert load_result.status == StageStatus.COMPLETED
        assert analyze_result.status == StageStatus.COMPLETED
        assert plan_result.status == StageStatus.COMPLETED
        assert transform_result.status == StageStatus.COMPLETED
        assert validate_result.status == StageStatus.COMPLETED
        assert export_result.status == StageStatus.COMPLETED

        artifact = context.get_artifact("artifact")
        assert artifact is not None
        assert artifact.compression_method == "int8"
        assert len(artifact.tensors) == 3
        assert artifact.overall_compression_ratio > 1.0

        validation = context.get_artifact("validation")
        assert validation.status == ValidationStatus.PASSED

    def test_full_pipeline_int4(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.INT4,
        )

        pipeline = Pipeline(config)
        register_default_stages(pipeline, config)

        context = pipeline.run(initial_input=str(model_path))

        assert context.get_stage_result(PipelineStage.LOAD).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.ANALYZE).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.PLAN).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.TRANSFORM).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.VALIDATE).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.EXPORT).status == StageStatus.COMPLETED

        artifact = context.get_artifact("artifact")
        assert artifact.compression_method == "int4"

    def test_full_pipeline_binary(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.BINARY,
        )

        pipeline = Pipeline(config)
        register_default_stages(pipeline, config)

        context = pipeline.run(initial_input=str(model_path))

        assert context.get_stage_result(PipelineStage.TRANSFORM).status == StageStatus.COMPLETED

        artifact = context.get_artifact("artifact")
        assert artifact.compression_method == "binary"
        assert artifact.overall_compression_ratio > 2.0

    def test_full_pipeline_ternary(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.TERNARY,
        )

        pipeline = Pipeline(config)
        register_default_stages(pipeline, config)

        context = pipeline.run(initial_input=str(model_path))

        assert context.get_stage_result(PipelineStage.TRANSFORM).status == StageStatus.COMPLETED

        artifact = context.get_artifact("artifact")
        assert artifact.compression_method == "ternary"

    def test_pipeline_with_skipped_validation(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.INT8,
            validation=ValidationConfig(enabled=False),
        )

        pipeline = Pipeline(config)
        register_default_stages(pipeline, config)

        context = pipeline.run(initial_input=str(model_path))

        validate_result = context.get_stage_result(PipelineStage.VALIDATE)
        assert validate_result.status == StageStatus.COMPLETED

        validation = context.get_artifact("validation")
        assert validation.status == ValidationStatus.SKIPPED

    def test_pipeline_with_max_size_constraint(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        output_path = tmp_path / "compiled"
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=output_path,
            compression_method=CompressionMethod.INT8,
            max_model_size_gb=0.000001,
        )

        pipeline = Pipeline(config)
        register_default_stages(pipeline, config)

        context = pipeline.run(initial_input=str(model_path))

        plan = context.get_artifact("plan")
        assert len(plan.warnings) > 0
        assert "exceeds" in plan.warnings[0]


class TestResultStructures:
    def test_compressed_tensor_info_serialization(self):
        from src.ldmark.compiler.result import CompressedTensorInfo

        info = CompressedTensorInfo(
            name="test.weight",
            original_shape=[100, 50],
            original_dtype="float32",
            original_size_bytes=20000,
            compression_method="int8",
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            compressed_size_bytes=5000,
            num_scales=391,
            scale_size_bytes=782,
            bits_per_weight=8.125,
            mae=0.01,
            mse=0.001,
            max_abs_error=0.1,
            relative_error=0.05,
        )

        data = info.to_dict()
        assert data["name"] == "test.weight"
        assert data["compression_method"] == "int8"

        restored = CompressedTensorInfo.from_dict(data)
        assert restored.name == info.name
        assert restored.compression_method == info.compression_method

    def test_compilation_artifact_serialization(self):
        from src.ldmark.compiler.result import CompressedTensorInfo

        artifact = CompilationArtifact(
            model_id="test_model",
            source_path="/path/to/model",
            compression_method="int8",
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            tensors=[
                CompressedTensorInfo(
                    name="weight",
                    original_shape=[100],
                    original_dtype="float32",
                    original_size_bytes=400,
                    compression_method="int8",
                    target_bits=8,
                    group_size=128,
                    scale_dtype="float16",
                    compressed_size_bytes=100,
                    num_scales=1,
                    scale_size_bytes=2,
                    bits_per_weight=8.125,
                )
            ],
            total_original_size_bytes=400,
            total_compressed_size_bytes=100,
            total_scale_size_bytes=2,
            overall_compression_ratio=4.0,
            overall_bits_per_weight=8.125,
        )

        data = artifact.to_dict()
        assert data["format_version"] == "ldmark-compilation-0.1"
        assert data["is_experimental"] is True
        assert data["compression_method"] == "int8"

        json_str = artifact.to_json()
        assert "ldmark-compilation-0.1" in json_str

        restored = CompilationArtifact.from_dict(data)
        assert restored.model_id == artifact.model_id
        assert restored.compression_method == artifact.compression_method


class TestErrorHandling:
    def test_invalid_compression_method(self, tmp_path):
        model_path = create_synthetic_npz(tmp_path / "model.npz")
        config = CompilerConfig(
            input_model_path=model_path,
            output_path=tmp_path / "out",
            compression_method=CompressionMethod.INT8,
        )

        load_stage = LoadStage(PipelineStage.LOAD)
        analyze_stage = AnalyzeStage(PipelineStage.ANALYZE)
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        loaded = load_stage.run(str(model_path), context)
        analyzed = analyze_stage.run(loaded, context)

        context.config.compression_method = "invalid"
        with pytest.raises(ValueError, match="Unsupported compression method"):
            plan_stage.run(analyzed, context)

    def test_missing_analysis(self, tmp_path):
        config = CompilerConfig(
            input_model_path=tmp_path / "model.npz",
            output_path=tmp_path / "out",
        )
        plan_stage = PlanStage(PipelineStage.PLAN)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        with pytest.raises(ValueError, match="No analysis available"):
            plan_stage.run({}, context)

    def test_missing_plan(self, tmp_path):
        config = CompilerConfig(
            input_model_path=tmp_path / "model.npz",
            output_path=tmp_path / "out",
        )
        transform_stage = TransformStage(PipelineStage.TRANSFORM)
        context = Pipeline.run.__globals__["PipelineContext"](config=config)

        with pytest.raises(ValueError, match="No compression plan available"):
            transform_stage.run({}, context)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])