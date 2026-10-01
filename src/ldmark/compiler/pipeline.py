"""
LDMARK Compilation Pipeline

Pipeline abstraction with independently replaceable stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar
import time

from .config import CompilerConfig, PipelineStage


T = TypeVar("T")
StageInput = TypeVar("StageInput")
StageOutput = TypeVar("StageOutput")


class StageStatus(Enum):
    """Status of a pipeline stage execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult(Generic[T]):
    """Result of a single pipeline stage."""
    stage: PipelineStage
    status: StageStatus
    output: Optional[T] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "output": str(self.output) if self.output is not None else None,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


class PipelineStageBase(ABC, Generic[StageInput, StageOutput]):
    """
    Base class for a compilation pipeline stage.

    Each stage is independently replaceable and testable.
    """

    def __init__(self, stage: PipelineStage, config: Optional[Dict[str, Any]] = None):
        self.stage = stage
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this stage."""
        pass

    @abstractmethod
    def run(self, input_data: StageInput, context: PipelineContext) -> StageOutput:
        """
        Execute the stage.

        Args:
            input_data: Input from previous stage (or initial input for first stage)
            context: Shared pipeline context with accumulated state

        Returns:
            Output data for next stage
        """
        pass

    def validate_input(self, input_data: StageInput) -> bool:
        """Optional input validation. Override if needed."""
        return True

    def validate_output(self, output_data: StageOutput) -> bool:
        """Optional output validation. Override if needed."""
        return True


@dataclass
class PipelineContext:
    """
    Shared context passed through all pipeline stages.

    Accumulates state as the pipeline progresses.
    """
    config: CompilerConfig
    stage_results: List[StageResult] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """Get an artifact from a previous stage."""
        return self.artifacts.get(key, default)

    def set_artifact(self, key: str, value: Any) -> None:
        """Store an artifact for later stages."""
        self.artifacts[key] = value

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def get_stage_result(self, stage: PipelineStage) -> Optional[StageResult]:
        """Get the result of a specific stage."""
        for result in self.stage_results:
            if result.stage == stage:
                return result
        return None

    def elapsed_time(self) -> float:
        """Get total elapsed time since pipeline start."""
        return time.time() - self.start_time


class Pipeline:
    """
    Compilation pipeline orchestrator.

    Manages stage execution, context passing, and result collection.
    """

    def __init__(self, config: CompilerConfig):
        self.config = config
        self.stages: Dict[PipelineStage, PipelineStageBase] = {}
        self._stage_order = list(PipelineStage)

    def register_stage(self, stage_impl: PipelineStageBase) -> None:
        """Register a stage implementation."""
        self.stages[stage_impl.stage] = stage_impl

    def unregister_stage(self, stage: PipelineStage) -> None:
        """Unregister a stage implementation."""
        self.stages.pop(stage, None)

    def get_stage(self, stage: PipelineStage) -> Optional[PipelineStageBase]:
        """Get a registered stage implementation."""
        return self.stages.get(stage)

    def run(self, initial_input: Any) -> PipelineContext:
        """
        Run the full compilation pipeline.

        Args:
            initial_input: Initial input data (e.g., model path)

        Returns:
            PipelineContext with all results and artifacts
        """
        context = PipelineContext(config=self.config)
        current_input = initial_input

        for stage_enum in self._stage_order:
            if not self.config.is_stage_enabled(stage_enum):
                result = StageResult(
                    stage=stage_enum,
                    status=StageStatus.SKIPPED,
                    metadata={"reason": "stage disabled in config"}
                )
                context.stage_results.append(result)
                continue

            stage_impl = self.stages.get(stage_enum)
            if stage_impl is None:
                result = StageResult(
                    stage=stage_enum,
                    status=StageStatus.SKIPPED,
                    metadata={"reason": f"No implementation registered for stage {stage_enum.value}"},
                )
                context.stage_results.append(result)
                continue

            # Validate input
            if not stage_impl.validate_input(current_input):
                result = StageResult(
                    stage=stage_enum,
                    status=StageStatus.FAILED,
                    error=f"Input validation failed for stage {stage_enum.value}",
                )
                context.stage_results.append(result)
                break

            # Run stage
            start_time = time.time()
            result = StageResult(stage=stage_enum, status=StageStatus.RUNNING)
            context.stage_results.append(result)

            try:
                output = stage_impl.run(current_input, context)
                duration = time.time() - start_time

                # Validate output
                if not stage_impl.validate_output(output):
                    result.status = StageStatus.FAILED
                    result.error = f"Output validation failed for stage {stage_enum.value}"
                    result.duration_seconds = duration
                    break

                result.status = StageStatus.COMPLETED
                result.output = output
                result.duration_seconds = duration
                current_input = output

            except Exception as e:
                result.status = StageStatus.FAILED
                result.error = f"{type(e).__name__}: {str(e)}"
                result.duration_seconds = time.time() - start_time
                context.add_warning(f"Stage {stage_enum.value} failed: {result.error}")
                break

        return context

    def run_stage(self, stage: PipelineStage, input_data: Any, context: PipelineContext) -> StageResult:
        """
        Run a single stage (useful for testing or incremental execution).

        Args:
            stage: Stage to run
            input_data: Input for this stage
            context: Pipeline context

        Returns:
            StageResult
        """
        stage_impl = self.stages.get(stage)
        if stage_impl is None:
            return StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error=f"No implementation registered for stage {stage.value}",
            )

        if not self.config.is_stage_enabled(stage):
            return StageResult(
                stage=stage,
                status=StageStatus.SKIPPED,
                metadata={"reason": "stage disabled in config"}
            )

        if not stage_impl.validate_input(input_data):
            return StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error=f"Input validation failed for stage {stage.value}",
            )

        start_time = time.time()
        result = StageResult(stage=stage, status=StageStatus.RUNNING)

        try:
            output = stage_impl.run(input_data, context)
            duration = time.time() - start_time

            if not stage_impl.validate_output(output):
                return StageResult(
                    stage=stage,
                    status=StageStatus.FAILED,
                    error=f"Output validation failed for stage {stage.value}",
                    duration_seconds=duration,
                )

            return StageResult(
                stage=stage,
                status=StageStatus.COMPLETED,
                output=output,
                duration_seconds=duration,
            )

        except Exception as e:
            return StageResult(
                stage=stage,
                status=StageStatus.FAILED,
                error=f"{type(e).__name__}: {str(e)}",
                duration_seconds=time.time() - start_time,
            )

    def get_results_summary(self, context: PipelineContext) -> Dict[str, Any]:
        """Get a summary of pipeline execution."""
        return {
            "total_duration_seconds": context.elapsed_time(),
            "stages": [r.to_dict() for r in context.stage_results],
            "warnings": context.warnings,
            "artifacts": list(context.artifacts.keys()),
        }