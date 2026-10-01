"""
LDMARK Compiler CLI

CLI interface for the compiler with real pipeline integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .compiler.config import (
    CompilerConfig,
    TargetHardware,
    PrecisionTarget,
    OptimizationStrategy,
    ValidationConfig,
    CompressionMethod,
)
from .compiler.pipeline import Pipeline, StageStatus
from .compiler.result import (
    CompilationResult,
    CompilationStatus,
    ModelReference,
    StorageMetrics,
    RuntimeMemoryMetrics,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ldmark",
        description="LDMARK Model Compiler - Laptop Designed Model Architecture for Reasoning and Knowledge",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="LDMARK Compiler 0.2.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a model without compiling")
    inspect_parser.add_argument("model", type=Path, help="Path to model file or directory")
    inspect_parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze model for compilation")
    analyze_parser.add_argument("model", type=Path, help="Path to model file or directory")
    analyze_parser.add_argument("--target-hardware", choices=[h.value for h in TargetHardware], default="unknown")
    analyze_parser.add_argument("--max-size-gb", type=float, help="Maximum model size in GB")
    analyze_parser.add_argument("--precision", choices=[p.value for p in PrecisionTarget], default="auto")
    analyze_parser.add_argument("--strategy", choices=[s.value for s in OptimizationStrategy], default="balanced")
    analyze_parser.add_argument("--output", type=Path, help="Output path for analysis results")
    analyze_parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")

    compile_parser = subparsers.add_parser("compile", help="Compile a model")
    compile_parser.add_argument("model", type=Path, help="Path to input model")
    compile_parser.add_argument("output", type=Path, help="Output path for compiled model")
    compile_parser.add_argument("--target-hardware", choices=[h.value for h in TargetHardware], default="unknown")
    compile_parser.add_argument("--max-size-gb", type=float, help="Maximum model size in GB")
    compile_parser.add_argument("--precision", choices=[p.value for p in PrecisionTarget], default="auto")
    compile_parser.add_argument("--strategy", choices=[s.value for s in OptimizationStrategy], default="balanced")
    compile_parser.add_argument("--method", choices=[m.value for m in CompressionMethod], default="auto")
    compile_parser.add_argument("--group-size", type=int, help="Force group size for quantization")
    compile_parser.add_argument("--skip-validation", action="store_true", help="Skip output validation")
    compile_parser.add_argument("--dry-run", action="store_true", help="Plan only, do not execute")
    compile_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    compile_parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark a compiled model")
    benchmark_parser.add_argument("model", type=Path, help="Path to compiled model")
    benchmark_parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    benchmark_parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    benchmark_parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")

    return parser


def cmd_inspect(args: argparse.Namespace) -> int:
    """Handle inspect command."""
    model_path = args.model

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    result = {
        "model_path": str(model_path),
        "exists": True,
        "is_directory": model_path.is_dir(),
        "size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        "format": model_path.suffix if model_path.is_file() else None,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Model: {model_path}")
        print(f"  Exists: {result['exists']}")
        print(f"  Is directory: {result['is_directory']}")
        if result['size_bytes'] is not None:
            print(f"  Size: {result['size_bytes']} bytes ({result['size_bytes'] / (1024**3):.4f} GB)")
        if result['format']:
            print(f"  Format: {result['format']}")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle analyze command with real analysis."""
    model_path = args.model

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    from ldmark.analysis.io import load_numpy_dict, load_safetensors_metadata, load_pytorch_state_dict
    from ldmark.analysis.analyzer import ModelAnalyzer

    suffix = model_path.suffix.lower()
    tensors = {}
    config = {}

    try:
        if suffix == ".safetensors":
            loaded = load_safetensors_metadata(model_path)
            tensors = loaded.get("tensors", {})
            config = loaded.get("metadata", {})
        elif suffix in (".pt", ".pth", ".bin"):
            state_dict = load_pytorch_state_dict(model_path)
            tensors = {k: v.cpu().numpy() if hasattr(v, "cpu") else v for k, v in state_dict.items()}
        elif suffix in (".npy", ".npz"):
            tensors = dict(load_numpy_dict(model_path))
        else:
            print(f"Error: Unsupported format for analysis: {suffix}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    if not tensors:
        print("Error: No tensors found in model", file=sys.stderr)
        return 1

    analyzer = ModelAnalyzer(model_id=model_path.stem, source_path=str(model_path))
    analysis = analyzer.analyze_numpy_tensors(tensors, config=config)

    result = {
        "model_id": analysis.model_id,
        "source_path": analysis.source_path,
        "parameter_count": analysis.parameter_counts.total,
        "num_tensors": len(analysis.tensors),
        "architecture": analysis.architecture.model_architecture,
        "num_layers": analysis.architecture.num_layers,
        "hidden_size": analysis.architecture.hidden_size,
        "storage_estimates": [e.to_dict() for e in analysis.storage_estimates],
        "compression_estimates": [e.to_dict() for e in analysis.compression_estimates[:5]],
        "runtime_memory": analysis.runtime_memory.to_dict() if analysis.runtime_memory else None,
        "warnings": analysis.warnings,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Analysis: {model_path}")
        print(f"  Parameters: {analysis.parameter_counts.total:,}")
        print(f"  Tensors: {len(analysis.tensors)}")
        print(f"  Architecture: {analysis.architecture.model_architecture or 'unknown'}")
        if analysis.architecture.num_layers:
            print(f"  Layers: {analysis.architecture.num_layers}")
        if analysis.architecture.hidden_size:
            print(f"  Hidden size: {analysis.architecture.hidden_size}")
        if analysis.storage_estimates:
            print(f"\n  Storage estimates:")
            for e in analysis.storage_estimates[:4]:
                print(f"    {e.dtype.value}: {e.estimated_gb:.4f} GB")
        if analysis.warnings:
            print(f"\n  Warnings:")
            for w in analysis.warnings:
                print(f"    - {w}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\n  Results written to: {args.output}")

    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    """Handle compile command with real pipeline."""
    model_path = args.model
    output_path = args.output

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    config = CompilerConfig(
        input_model_path=model_path,
        output_path=output_path,
        target_hardware=TargetHardware(args.target_hardware),
        max_model_size_gb=args.max_size_gb,
        target_precision=PrecisionTarget(args.precision),
        optimization_strategy=OptimizationStrategy(args.strategy),
        compression_method=CompressionMethod(args.method),
        force_group_size=args.group_size,
        validation=ValidationConfig(enabled=not args.skip_validation),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    from .compiler.stages import register_default_stages

    pipeline = Pipeline(config)
    register_default_stages(pipeline, config)

    print(f"LDMARK Compiler")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Method: {config.compression_method.value}")
    print(f"  Strategy: {config.optimization_strategy.value}")
    print()

    context = pipeline.run(initial_input=str(model_path))

    transform_result = context.get_artifact("transform_result")
    export_result = context.get_artifact("export_result")
    artifact = context.get_artifact("artifact")
    validation = context.get_artifact("validation")

    storage = StorageMetrics()
    if transform_result:
        total_orig = transform_result.get("total_original_size_bytes", 0)
        total_comp = transform_result.get("total_compressed_size_bytes", 0)
        total_scales = transform_result.get("total_scale_size_bytes", 0)
        actual_comp = total_comp + total_scales
        storage = StorageMetrics(
            original_size_bytes=total_orig,
            compressed_size_bytes=actual_comp,
            compression_ratio=total_orig / actual_comp if actual_comp > 0 else None,
            original_format=model_path.suffix,
            compressed_format="ldmark-compilation-0.1",
        )

    runtime = RuntimeMemoryMetrics()
    if artifact and artifact.parameter_count > 0:
        weight_bytes = artifact.total_compressed_size_bytes
        runtime = RuntimeMemoryMetrics(
            weights_gb=weight_bytes / (1024 ** 3),
            assumptions={"note": "Runtime memory is estimated from compressed weights only"},
        )

    result = CompilationResult(
        status=CompilationStatus.SUCCESS,
        config_snapshot=config.to_dict(),
        input_model=ModelReference(
            path=model_path,
            size_bytes=model_path.stat().st_size if model_path.is_file() else None,
        ),
        output_model=ModelReference(path=output_path) if export_result else None,
        storage=storage,
        runtime_memory=runtime,
        selected_strategy=config.optimization_strategy.value,
        artifact=artifact,
        validation=validation,
        pipeline_stages=[r.to_dict() for r in context.stage_results],
        warnings=context.warnings,
    )
    result.mark_completed()

    if args.format == "json":
        print(result.to_json())
    else:
        print(f"Compilation {'planned' if config.dry_run else 'completed'}")
        print(f"  Status: {result.status.value}")
        print(f"  Duration: {result.duration_seconds:.3f}s")

        completed = [r for r in context.stage_results if r.status == StageStatus.COMPLETED]
        failed = [r for r in context.stage_results if r.status == StageStatus.FAILED]
        print(f"  Stages completed: {len(completed)}/{len(context.stage_results)}")

        if artifact:
            print(f"\n  Compression Results:")
            print(f"    Method: {artifact.compression_method}")
            print(f"    Original size: {artifact.total_original_size_bytes:,} bytes")
            print(f"    Compressed size: {artifact.total_compressed_size_bytes + artifact.total_scale_size_bytes:,} bytes")
            print(f"    Compression ratio: {artifact.overall_compression_ratio:.2f}x")
            print(f"    Bits per weight: {artifact.overall_bits_per_weight:.3f}")
            print(f"    Tensors compressed: {len(artifact.tensors)}")

        if validation:
            print(f"\n  Validation: {validation.status.value}")
            if validation.details:
                print(f"    Max MAE: {validation.details.get('max_mae', 'N/A')}")
                print(f"    Max relative error: {validation.details.get('max_relative_error', 'N/A')}")

        if context.warnings:
            print(f"\n  Warnings:")
            for w in context.warnings:
                print(f"    - {w}")

        if failed:
            print(f"\n  Failed stages:")
            for r in failed:
                print(f"    - {r.stage.value}: {r.error}")

    return 0 if result.status != CompilationStatus.FAILED else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle benchmark command."""
    model_path = args.model

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        return 1

    result = {
        "model_path": str(model_path),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "note": "Benchmark not yet implemented - requires runtime inference engine",
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Benchmark: {model_path}")
        print(f"  Iterations: {args.iterations}")
        print(f"  Warmup: {args.warmup}")
        print(f"\n  Note: {result['note']}")

    return 0


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "inspect":
            return cmd_inspect(args)
        elif args.command == "analyze":
            return cmd_analyze(args)
        elif args.command == "compile":
            return cmd_compile(args)
        elif args.command == "benchmark":
            return cmd_benchmark(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())