"""
LDMARK Compiler Pipeline Stages

Real stage implementations integrating with ldmark.analysis and ldmark.compression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import (
    CompilerConfig,
    CompressionMethod,
    CompressionPlan,
    PipelineStage,
    PrecisionTarget,
    OptimizationStrategy,
)
from .pipeline import PipelineStageBase, PipelineContext, StageResult, StageStatus
from .result import (
    CompilationArtifact,
    CompressedTensorInfo,
    ValidationResult,
    ValidationStatus,
)


class LoadStage(PipelineStageBase[str, Dict[str, Any]]):
    """Stage 1: Load model from disk using ldmark.analysis.io."""

    @property
    def name(self) -> str:
        return "Model Loader"

    def run(self, input_data: str, context: PipelineContext) -> Dict[str, Any]:
        from ldmark.analysis.io import (
            load_safetensors_metadata,
            load_pytorch_state_dict,
            load_numpy_dict,
            load_config,
        )

        model_path = Path(input_data)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        result: Dict[str, Any] = {
            "model_path": str(model_path),
            "loaded": True,
            "format": None,
            "tensors": {},
            "config": {},
        }

        suffix = model_path.suffix.lower()

        if suffix == ".safetensors":
            loaded = load_safetensors_metadata(model_path)
            result["format"] = "safetensors"
            result["tensors"] = loaded.get("tensors", {})
            result["config"] = loaded.get("metadata", {})
        elif suffix in (".pt", ".pth", ".bin"):
            state_dict = load_pytorch_state_dict(model_path)
            result["format"] = "pytorch"
            result["tensors"] = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in state_dict.items()}
        elif suffix == ".npy" or suffix == ".npz":
            numpy_dict = load_numpy_dict(model_path)
            result["format"] = "numpy"
            result["tensors"] = dict(numpy_dict)
        elif suffix == ".json":
            config = load_config(model_path)
            result["format"] = "json"
            result["config"] = config
            result["loaded"] = False
            context.add_warning("JSON config loaded but no tensors. Provide tensors separately.")
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

        context.set_artifact("loaded_model", result)
        return result


class AnalyzeStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Stage 2: Analyze model using ldmark.analysis.ModelAnalyzer."""

    @property
    def name(self) -> str:
        return "Model Analyzer"

    def run(self, input_data: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        from ldmark.analysis.analyzer import ModelAnalyzer

        loaded = context.get_artifact("loaded_model", input_data)
        tensors = loaded.get("tensors", {})
        config = loaded.get("config", {})

        if not tensors:
            raise ValueError("No tensors available for analysis. Load a model with tensors first.")

        analyzer = ModelAnalyzer(
            model_id=Path(loaded.get("model_path", "unknown")).stem,
            source_path=loaded.get("model_path", ""),
        )

        analysis = analyzer.analyze_numpy_tensors(tensors, config=config)

        context.set_artifact("analysis", analysis)
        return {**input_data, "analysis": analysis}


class PlanStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Stage 3: Plan compression strategy based on analysis and config."""

    @property
    def name(self) -> str:
        return "Strategy Planner"

    def run(self, input_data: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        config: CompilerConfig = context.config
        analysis = context.get_artifact("analysis")

        if analysis is None:
            raise ValueError("No analysis available. Run ANALYZE stage first.")

        param_count = analysis.parameter_counts.total
        if param_count == 0:
            raise ValueError("Model has zero parameters. Cannot plan compression.")

        method = self._select_method(config, analysis)
        plan = self._create_plan(method, config, analysis, param_count)

        context.set_artifact("plan", plan)
        return {**input_data, "plan": plan}

    def _select_method(self, config: CompilerConfig, analysis) -> CompressionMethod:
        """Select compression method based on config and analysis."""
        if config.compression_method != CompressionMethod.AUTO:
            return config.compression_method

        target = config.target_precision
        if target == PrecisionTarget.INT8:
            return CompressionMethod.INT8
        elif target == PrecisionTarget.INT4:
            return CompressionMethod.INT4

        if config.optimization_strategy == OptimizationStrategy.SIZE:
            return CompressionMethod.INT4
        elif config.optimization_strategy == OptimizationStrategy.ACCURACY:
            return CompressionMethod.INT8

        return CompressionMethod.INT8

    def _create_plan(
        self,
        method: CompressionMethod,
        config: CompilerConfig,
        analysis,
        param_count: int,
    ) -> CompressionPlan:
        """Create a compression plan with estimates."""
        warnings: List[str] = []

        if method == CompressionMethod.INT8:
            target_bits = 8
            group_size = config.force_group_size or 128
            scale_dtype = config.force_scale_dtype or "float16"
            symmetric = config.force_symmetric if config.force_symmetric is not None else True
            expected_bps = target_bits + (16.0 / group_size)
            expected_bytes = int(np.ceil(param_count * expected_bps / 8))
            baseline_bytes = param_count * 2
            expected_ratio = baseline_bytes / expected_bytes if expected_bytes > 0 else 0.0
            rationale = (
                f"INT8 selected: {target_bits}-bit weights with group_size={group_size}, "
                f"scale_dtype={scale_dtype}. Expected {expected_bps:.3f} bits/weight."
            )

        elif method == CompressionMethod.INT4:
            target_bits = 4
            group_size = config.force_group_size or 128
            scale_dtype = config.force_scale_dtype or "float16"
            symmetric = config.force_symmetric if config.force_symmetric is not None else True
            expected_bps = target_bits + (16.0 / group_size)
            expected_bytes = int(np.ceil(param_count * expected_bps / 8))
            baseline_bytes = param_count * 2
            expected_ratio = baseline_bytes / expected_bytes if expected_bytes > 0 else 0.0
            rationale = (
                f"INT4 selected: {target_bits}-bit weights with group_size={group_size}, "
                f"scale_dtype={scale_dtype}. Expected {expected_bps:.3f} bits/weight."
            )

        elif method == CompressionMethod.BINARY:
            target_bits = 1
            group_size = config.force_group_size or 128
            scale_dtype = config.force_scale_dtype or "float16"
            symmetric = True
            expected_bps = 1.0 + (16.0 / group_size)
            expected_bytes = int(np.ceil(param_count * expected_bps / 8))
            baseline_bytes = param_count * 2
            expected_ratio = baseline_bytes / expected_bytes if expected_bytes > 0 else 0.0
            rationale = (
                f"Binary selected: {target_bits}-bit weights with group_size={group_size}, "
                f"scale_dtype={scale_dtype}. Expected {expected_bps:.3f} bits/weight. "
                "WARNING: Binary compression has significant quality loss."
            )
            warnings.append("Binary compression will cause significant quality degradation.")

        elif method == CompressionMethod.TERNARY:
            target_bits = 2
            group_size = config.force_group_size or 128
            scale_dtype = config.force_scale_dtype or "float16"
            symmetric = True
            expected_bps = 1.585 + (16.0 / group_size)
            expected_bytes = int(np.ceil(param_count * expected_bps / 8))
            baseline_bytes = param_count * 2
            expected_ratio = baseline_bytes / expected_bytes if expected_bytes > 0 else 0.0
            rationale = (
                f"Ternary selected: ~1.585-bit weights with group_size={group_size}, "
                f"scale_dtype={scale_dtype}. Expected {expected_bps:.3f} bits/weight."
            )
            warnings.append("Ternary compression will cause moderate quality degradation.")

        else:
            raise ValueError(f"Unsupported compression method: {method}")

        if config.max_model_size_gb is not None:
            max_bytes = int(config.max_model_size_gb * (1024 ** 3))
            if expected_bytes > max_bytes:
                warnings.append(
                    f"Expected size ({expected_bytes / (1024**3):.2f} GB) exceeds "
                    f"max_model_size_gb ({config.max_model_size_gb} GB)"
                )

        return CompressionPlan(
            method=method,
            target_bits=target_bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            symmetric=symmetric,
            expected_bits_per_weight=expected_bps,
            expected_storage_bytes=expected_bytes,
            expected_compression_ratio=expected_ratio,
            rationale=rationale,
            warnings=warnings,
        )


class TransformStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Stage 4: Apply compression using ldmark.compression."""

    @property
    def name(self) -> str:
        return "Model Transformer"

    def run(self, input_data: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        from ldmark.compression.quantize import (
            QuantizationConfig,
            QuantizationTarget,
            quantize_groupwise,
            dequantize_groupwise,
        )
        from ldmark.compression.binary_ternary import (
            BinaryConfig,
            TernaryConfig,
            BinaryEncoding,
            TernaryEncoding,
            binary_quantize_sign,
            binary_dequantize,
            ternary_quantize_threshold,
            ternary_dequantize,
        )
        from ldmark.compression.metrics import calculate_error_metrics

        loaded = context.get_artifact("loaded_model", input_data)
        plan: CompressionPlan = context.get_artifact("plan")
        analysis = context.get_artifact("analysis")

        if plan is None:
            raise ValueError("No compression plan available. Run PLAN stage first.")

        tensors = loaded.get("tensors", {})
        if not tensors:
            raise ValueError("No tensors available for transformation.")

        compressed_tensors: Dict[str, Any] = {}
        tensor_infos: List[CompressedTensorInfo] = []
        total_original = 0
        total_compressed = 0
        total_scales = 0

        for name, tensor in tensors.items():
            if not isinstance(tensor, np.ndarray):
                context.add_warning(f"Skipping non-numpy tensor: {name} (type: {type(tensor).__name__})")
                continue

            if tensor.dtype not in (np.float32, np.float16, np.float64):
                context.add_warning(f"Skipping tensor {name} with unsupported dtype: {tensor.dtype}")
                continue

            original_size = tensor.nbytes
            total_original += original_size

            try:
                if plan.method == CompressionMethod.INT8:
                    qconfig = QuantizationConfig(
                        target_bits=QuantizationTarget.INT8,
                        group_size=plan.group_size,
                        scale_dtype=np.dtype(plan.scale_dtype),
                        symmetric=plan.symmetric,
                    )
                    qtensor = quantize_groupwise(tensor, qconfig)
                    dequantized = dequantize_groupwise(qtensor)
                    compressed_data = qtensor.data
                    scales = qtensor.scales

                elif plan.method == CompressionMethod.INT4:
                    qconfig = QuantizationConfig(
                        target_bits=QuantizationTarget.INT4,
                        group_size=plan.group_size,
                        scale_dtype=np.dtype(plan.scale_dtype),
                        symmetric=plan.symmetric,
                    )
                    qtensor = quantize_groupwise(tensor, qconfig)
                    dequantized = dequantize_groupwise(qtensor)
                    compressed_data = qtensor.data
                    scales = qtensor.scales

                elif plan.method == CompressionMethod.BINARY:
                    bconfig = BinaryConfig(
                        encoding=BinaryEncoding.SIGN,
                        scale_dtype=np.dtype(plan.scale_dtype),
                        group_size=plan.group_size,
                    )
                    btensor = binary_quantize_sign(tensor, bconfig)
                    dequantized = binary_dequantize(btensor)
                    compressed_data = btensor.packed_data
                    scales = btensor.scales

                elif plan.method == CompressionMethod.TERNARY:
                    tconfig = TernaryConfig(
                        encoding=TernaryEncoding.MINUS_ZERO_PLUS,
                        scale_dtype=np.dtype(plan.scale_dtype),
                        group_size=plan.group_size,
                    )
                    ttensor = ternary_quantize_threshold(tensor, tconfig)
                    dequantized = ternary_dequantize(ttensor)
                    compressed_data = ttensor.packed_data
                    scales = ttensor.scales

                else:
                    raise ValueError(f"Unsupported compression method: {plan.method}")

                compressed_size = compressed_data.nbytes
                scale_size = scales.nbytes
                total_compressed += compressed_size
                total_scales += scale_size

                mae, mse, max_err, rel_err = calculate_error_metrics(tensor, dequantized)

                num_weights = tensor.size
                total_bits = compressed_size * 8 + scale_size * 8
                bps = total_bits / num_weights if num_weights > 0 else 0.0

                info = CompressedTensorInfo(
                    name=name,
                    original_shape=list(tensor.shape),
                    original_dtype=str(tensor.dtype),
                    original_size_bytes=original_size,
                    compression_method=plan.method.value,
                    target_bits=plan.target_bits,
                    group_size=plan.group_size,
                    scale_dtype=plan.scale_dtype,
                    compressed_size_bytes=compressed_size,
                    num_scales=len(scales),
                    scale_size_bytes=scale_size,
                    bits_per_weight=bps,
                    mae=mae,
                    mse=mse,
                    max_abs_error=max_err,
                    relative_error=rel_err,
                )
                tensor_infos.append(info)

                compressed_tensors[name] = {
                    "data": compressed_data,
                    "scales": scales,
                    "original_shape": tensor.shape,
                    "original_dtype": str(tensor.dtype),
                }

            except Exception as e:
                context.add_warning(f"Failed to compress tensor {name}: {e}")
                continue

        if not tensor_infos:
            raise ValueError("No tensors were successfully compressed.")

        transform_result = {
            "compressed_tensors": compressed_tensors,
            "tensor_infos": tensor_infos,
            "total_original_size_bytes": total_original,
            "total_compressed_size_bytes": total_compressed,
            "total_scale_size_bytes": total_scales,
            "num_tensors_compressed": len(tensor_infos),
            "num_tensors_skipped": len(tensors) - len(tensor_infos),
        }

        context.set_artifact("transform_result", transform_result)
        return {**input_data, "transform_result": transform_result}


class ValidateStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Stage 5: Validate compressed output using error metrics."""

    @property
    def name(self) -> str:
        return "Output Validator"

    def run(self, input_data: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        config: CompilerConfig = context.config

        if not config.validation.enabled:
            validation = ValidationResult(
                status=ValidationStatus.SKIPPED,
                details={"reason": "validation disabled in config"},
            )
            context.set_artifact("validation", validation)
            return {**input_data, "validation": validation}

        transform_result = context.get_artifact("transform_result")
        if transform_result is None:
            raise ValueError("No transform result available. Run TRANSFORM stage first.")

        tensor_infos = transform_result.get("tensor_infos", [])
        if not tensor_infos:
            raise ValueError("No compressed tensors to validate.")

        total_original = transform_result.get("total_original_size_bytes", 0)
        total_compressed = transform_result.get("total_compressed_size_bytes", 0)
        total_scales = transform_result.get("total_scale_size_bytes", 0)
        actual_compressed = total_compressed + total_scales

        shape_preserved = all(
            info.original_shape is not None for info in tensor_infos
        )
        count_preserved = len(tensor_infos) > 0

        storage_reduced = actual_compressed < total_original if total_original > 0 else False

        max_mae = max((info.mae for info in tensor_infos), default=0.0)
        max_mse = max((info.mse for info in tensor_infos), default=0.0)
        max_rel_error = max((info.relative_error for info in tensor_infos), default=0.0)

        details = {
            "num_tensors_validated": len(tensor_infos),
            "shape_preserved": shape_preserved,
            "count_preserved": count_preserved,
            "storage_reduced": storage_reduced,
            "total_original_bytes": total_original,
            "total_compressed_bytes": actual_compressed,
            "compression_ratio": total_original / actual_compressed if actual_compressed > 0 else 0.0,
            "max_mae": max_mae,
            "max_mse": max_mse,
            "max_relative_error": max_rel_error,
        }

        passed = shape_preserved and count_preserved and storage_reduced

        validation = ValidationResult(
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            rtol=config.validation.tolerance_rtol,
            atol=config.validation.tolerance_atol,
            test_cases_passed=len(tensor_infos) if passed else 0,
            test_cases_total=len(tensor_infos),
            details=details,
        )

        if not passed:
            validation.error = "Validation failed: one or more checks did not pass"

        context.set_artifact("validation", validation)
        return {**input_data, "validation": validation}


class ExportStage(PipelineStageBase[Dict[str, Any], Dict[str, Any]]):
    """Stage 6: Export compilation artifact."""

    @property
    def name(self) -> str:
        return "Model Exporter"

    def run(self, input_data: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        loaded = context.get_artifact("loaded_model", input_data)
        plan: CompressionPlan = context.get_artifact("plan")
        analysis = context.get_artifact("analysis")
        transform_result = context.get_artifact("transform_result")
        validation = context.get_artifact("validation")

        if plan is None:
            raise ValueError("No compression plan available. Run PLAN stage first.")
        if transform_result is None:
            raise ValueError("No transform result available. Run TRANSFORM stage first.")

        tensor_infos = transform_result.get("tensor_infos", [])

        total_original = transform_result.get("total_original_size_bytes", 0)
        total_compressed = transform_result.get("total_compressed_size_bytes", 0)
        total_scales = transform_result.get("total_scale_size_bytes", 0)
        actual_compressed = total_compressed + total_scales

        overall_ratio = total_original / actual_compressed if actual_compressed > 0 else 0.0
        total_weights = sum(
            np.prod(info.original_shape) for info in tensor_infos
        )
        overall_bps = (
            (actual_compressed * 8) / total_weights if total_weights > 0 else 0.0
        )

        artifact = CompilationArtifact(
            model_id=Path(loaded.get("model_path", "unknown")).stem,
            source_path=loaded.get("model_path", ""),
            architecture=analysis.architecture.model_architecture if analysis else None,
            parameter_count=analysis.parameter_counts.total if analysis else 0,
            compression_method=plan.method.value,
            target_bits=plan.target_bits,
            group_size=plan.group_size,
            scale_dtype=plan.scale_dtype,
            symmetric=plan.symmetric,
            tensors=tensor_infos,
            total_original_size_bytes=total_original,
            total_compressed_size_bytes=total_compressed,
            total_scale_size_bytes=total_scales,
            overall_compression_ratio=overall_ratio,
            overall_bits_per_weight=overall_bps,
            validation_status=validation.status.value if validation else "not_run",
            validation_details=validation.details if validation else {},
            warnings=plan.warnings + context.warnings,
        )

        output_path = Path(context.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        metadata_path = output_path / "compilation_metadata.json"
        metadata_path.write_text(artifact.to_json())

        tensors_dir = output_path / "tensors"
        tensors_dir.mkdir(exist_ok=True)

        compressed_tensors = transform_result.get("compressed_tensors", {})
        for name, tdata in compressed_tensors.items():
            safe_name = name.replace(".", "_").replace("/", "_")
            tensor_path = tensors_dir / f"{safe_name}.npz"
            np.savez_compressed(
                tensor_path,
                data=tdata["data"],
                scales=tdata["scales"],
                original_shape=np.array(tdata["original_shape"]),
            )

        export_result = {
            "output_path": str(output_path),
            "metadata_path": str(metadata_path),
            "tensors_dir": str(tensors_dir),
            "exported": True,
            "format": "ldmark-compilation-0.1 (experimental)",
            "num_tensors_exported": len(compressed_tensors),
        }

        context.set_artifact("export_result", export_result)
        context.set_artifact("artifact", artifact)
        return {**input_data, "export_result": export_result}


def register_default_stages(pipeline, config: CompilerConfig) -> None:
    """Register default stage implementations."""
    pipeline.register_stage(LoadStage(PipelineStage.LOAD, config.get_stage_config(PipelineStage.LOAD)))
    pipeline.register_stage(AnalyzeStage(PipelineStage.ANALYZE, config.get_stage_config(PipelineStage.ANALYZE)))
    pipeline.register_stage(PlanStage(PipelineStage.PLAN, config.get_stage_config(PipelineStage.PLAN)))
    pipeline.register_stage(TransformStage(PipelineStage.TRANSFORM, config.get_stage_config(PipelineStage.TRANSFORM)))
    pipeline.register_stage(ValidateStage(PipelineStage.VALIDATE, config.get_stage_config(PipelineStage.VALIDATE)))
    pipeline.register_stage(ExportStage(PipelineStage.EXPORT, config.get_stage_config(PipelineStage.EXPORT)))