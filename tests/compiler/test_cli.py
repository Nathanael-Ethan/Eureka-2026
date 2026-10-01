"""
Tests for LDMARK CLI
"""

from __future__ import annotations

from pathlib import Path
import sys
import pytest

from src.ldmark.cli import main, create_parser


class TestCLIParser:
    def test_parser_creation(self):
        parser = create_parser()
        assert parser is not None

    def test_inspect_command(self):
        parser = create_parser()
        args = parser.parse_args(["inspect", "/models/test"])
        assert args.command == "inspect"
        assert args.model == Path("/models/test")

    def test_analyze_command(self):
        parser = create_parser()
        args = parser.parse_args([
            "analyze", "/models/test",
            "--target-hardware", "laptop_cpu",
            "--max-size-gb", "4.0",
            "--precision", "int4",
        ])
        assert args.command == "analyze"
        assert args.target_hardware == "laptop_cpu"
        assert args.max_size_gb == 4.0
        assert args.precision == "int4"

    def test_compile_command(self):
        parser = create_parser()
        args = parser.parse_args([
            "compile", "/models/input", "/models/output",
            "--strategy", "size",
            "--dry-run",
        ])
        assert args.command == "compile"
        assert args.model == Path("/models/input")
        assert args.output == Path("/models/output")
        assert args.strategy == "size"
        assert args.dry_run is True

    def test_benchmark_command(self):
        parser = create_parser()
        args = parser.parse_args([
            "benchmark", "/models/compiled",
            "--iterations", "20",
        ])
        assert args.command == "benchmark"
        assert args.iterations == 20


class TestCLICommands:
    def test_inspect_nonexistent(self, capsys):
        exit_code = main(["inspect", "/nonexistent/path"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error: Model path does not exist" in captured.err

    def test_inspect_existing_file(self, tmp_path, capsys):
        test_file = tmp_path / "model.bin"
        test_file.write_text("dummy")

        exit_code = main(["inspect", str(test_file)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "Exists: True" in captured.out

    def test_analyze_nonexistent(self, capsys):
        exit_code = main(["analyze", "/nonexistent/path"])
        assert exit_code == 1

    def test_analyze_numpy_model(self, tmp_path, capsys):
        import numpy as np

        model_path = tmp_path / "model.npz"
        np.savez(model_path, weight=np.random.randn(100, 50).astype(np.float32))

        exit_code = main(["analyze", str(model_path)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Analysis:" in captured.out
        assert "Parameters:" in captured.out

    def test_compile_nonexistent(self, capsys):
        exit_code = main(["compile", "/nonexistent/path", "/output/path"])
        assert exit_code == 1

    def test_compile_numpy_model(self, tmp_path, capsys):
        import numpy as np

        model_path = tmp_path / "model.npz"
        np.savez(model_path, weight=np.random.randn(100, 50).astype(np.float32))
        output_path = tmp_path / "output"

        exit_code = main(["compile", str(model_path), str(output_path), "--method", "int8"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Compilation completed" in captured.out
        assert "Compression ratio:" in captured.out

    def test_compile_with_skip_validation(self, tmp_path, capsys):
        import numpy as np

        model_path = tmp_path / "model.npz"
        np.savez(model_path, weight=np.random.randn(100, 50).astype(np.float32))
        output_path = tmp_path / "output"

        exit_code = main(["compile", str(model_path), str(output_path), "--skip-validation", "--method", "int8"])
        assert exit_code == 0

    def test_benchmark_nonexistent(self, capsys):
        exit_code = main(["benchmark", "/nonexistent/path"])
        assert exit_code == 1

    def test_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "LDMARK Model Compiler" in captured.out
        assert "inspect" in captured.out
        assert "analyze" in captured.out
        assert "compile" in captured.out
        assert "benchmark" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])