"""
Tests for LDMARK Compiler Pipeline
"""

from __future__ import annotations

from typing import Any, Dict
import pytest

from src.ldmark.compiler.config import CompilerConfig, PipelineStage
from src.ldmark.compiler.pipeline import Pipeline, PipelineContext, PipelineStageBase, StageResult, StageStatus


class DummyStage(PipelineStageBase[str, Dict[str, Any]]):
    """Test stage that can be configured to succeed or fail."""

    def __init__(
        self,
        stage: PipelineStage,
        config: Dict[str, Any] = None,
        should_fail: bool = False,
        fail_message: str = "Stage failed",
        output_data: Dict[str, Any] = None,
    ):
        super().__init__(stage, config)
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.output_data = output_data or {"result": "success"}

    @property
    def name(self) -> str:
        return f"Dummy {self.stage.value}"

    def run(self, input_data: str, context: PipelineContext) -> Dict[str, Any]:
        if self.should_fail:
            raise RuntimeError(self.fail_message)
        return {**self.output_data, "input_was": input_data}


class TestPipelineStageBase:
    def test_stage_creation(self):
        stage = DummyStage(PipelineStage.LOAD)
        assert stage.stage == PipelineStage.LOAD
        assert stage.name == "Dummy load"

    def test_stage_run_success(self):
        stage = DummyStage(PipelineStage.LOAD)
        context = PipelineContext(config=CompilerConfig(input_model_path="test", output_path="out"))
        result = stage.run("test_input", context)
        assert result["result"] == "success"
        assert result["input_was"] == "test_input"

    def test_stage_run_failure(self):
        stage = DummyStage(PipelineStage.LOAD, should_fail=True, fail_message="Test error")
        context = PipelineContext(config=CompilerConfig(input_model_path="test", output_path="out"))
        with pytest.raises(RuntimeError, match="Test error"):
            stage.run("test_input", context)


class TestPipelineContext:
    def test_artifact_storage(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        context = PipelineContext(config=config)

        context.set_artifact("key1", "value1")
        assert context.get_artifact("key1") == "value1"
        assert context.get_artifact("missing", "default") == "default"

    def test_warnings(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        context = PipelineContext(config=config)

        context.add_warning("Warning 1")
        context.add_warning("Warning 2")
        assert len(context.warnings) == 2
        assert "Warning 1" in context.warnings

    def test_stage_results(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        context = PipelineContext(config=config)

        result1 = StageResult(stage=PipelineStage.LOAD, status=StageStatus.COMPLETED)
        result2 = StageResult(stage=PipelineStage.ANALYZE, status=StageStatus.COMPLETED)
        context.stage_results.extend([result1, result2])

        assert context.get_stage_result(PipelineStage.LOAD) == result1
        assert context.get_stage_result(PipelineStage.ANALYZE) == result2
        assert context.get_stage_result(PipelineStage.PLAN) is None


class TestPipeline:
    def test_pipeline_creation(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)
        assert pipeline.config == config

    def test_register_stage(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)
        stage = DummyStage(PipelineStage.LOAD)

        pipeline.register_stage(stage)
        assert pipeline.get_stage(PipelineStage.LOAD) == stage

    def test_unregister_stage(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)
        stage = DummyStage(PipelineStage.LOAD)

        pipeline.register_stage(stage)
        pipeline.unregister_stage(PipelineStage.LOAD)
        assert pipeline.get_stage(PipelineStage.LOAD) is None

    def test_run_success(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)

        pipeline.register_stage(DummyStage(PipelineStage.LOAD, output_data={"stage": "load"}))
        pipeline.register_stage(DummyStage(PipelineStage.ANALYZE, output_data={"stage": "analyze"}))
        pipeline.register_stage(DummyStage(PipelineStage.PLAN, output_data={"stage": "plan"}))

        context = pipeline.run("initial_input")

        assert context.get_stage_result(PipelineStage.LOAD).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.ANALYZE).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.PLAN).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.TRANSFORM).status == StageStatus.SKIPPED

    def test_run_stage_failure_stops_pipeline(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)

        pipeline.register_stage(DummyStage(PipelineStage.LOAD))
        pipeline.register_stage(DummyStage(PipelineStage.ANALYZE, should_fail=True))
        pipeline.register_stage(DummyStage(PipelineStage.PLAN))

        context = pipeline.run("initial_input")

        assert context.get_stage_result(PipelineStage.LOAD).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.ANALYZE).status == StageStatus.FAILED
        # PLAN stage should not run after ANALYZE failure - get_stage_result returns None
        plan_result = context.get_stage_result(PipelineStage.PLAN)
        assert plan_result is None

    def test_run_disabled_stage(self):
        config = CompilerConfig(
            input_model_path="test",
            output_path="out",
            skip_stages=[PipelineStage.ANALYZE],
        )
        pipeline = Pipeline(config)

        pipeline.register_stage(DummyStage(PipelineStage.LOAD))
        pipeline.register_stage(DummyStage(PipelineStage.ANALYZE))

        context = pipeline.run("initial_input")

        assert context.get_stage_result(PipelineStage.LOAD).status == StageStatus.COMPLETED
        assert context.get_stage_result(PipelineStage.ANALYZE).status == StageStatus.SKIPPED

    def test_run_missing_stage_implementation(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)
        # Don't register LOAD stage

        context = pipeline.run("initial_input")

        assert context.get_stage_result(PipelineStage.LOAD).status == StageStatus.SKIPPED
        assert "No implementation registered" in context.get_stage_result(PipelineStage.LOAD).metadata.get("reason", "")

    def test_run_stage_method(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)
        stage = DummyStage(PipelineStage.LOAD, output_data={"custom": "data"})
        pipeline.register_stage(stage)

        context = PipelineContext(config=config)
        result = pipeline.run_stage(PipelineStage.LOAD, "test_input", context)

        assert result.status == StageStatus.COMPLETED
        assert result.output["custom"] == "data"

    def test_results_summary(self):
        config = CompilerConfig(input_model_path="test", output_path="out")
        pipeline = Pipeline(config)

        pipeline.register_stage(DummyStage(PipelineStage.LOAD))
        pipeline.register_stage(DummyStage(PipelineStage.ANALYZE, should_fail=True))

        context = pipeline.run("initial_input")
        summary = pipeline.get_results_summary(context)

        assert "total_duration_seconds" in summary
        assert len(summary["stages"]) >= 2
        assert summary["stages"][0]["stage"] == "load"
        assert summary["stages"][1]["stage"] == "analyze"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])