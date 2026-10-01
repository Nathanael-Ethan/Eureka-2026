# LDMARK

**Laptop Designed Model Architecture for Reasoning and Knowledge**

A model compilation and compression system for analyzing, planning, and transforming neural network models. Built for the Eureka 2026 Hackathon.

## Overview

LDMARK is a Python-based toolkit that provides a complete pipeline for model compression — from loading and analyzing models to planning and executing quantization strategies. It supports multiple model formats, hardware profiles, and compression methods with a focus on factual feasibility analysis.

## Features

- **Multi-format model loading** — PyTorch (`.pt`, `.pth`, `.bin`), safetensors, NumPy (`.npy`, `.npz`)
- **Model analysis** — Parameter counting, architecture detection, storage estimation, runtime memory profiling
- **Compression planning** — Constraint-based planning with hardware-aware recommendations
- **Quantization** — INT8, INT4, binary, and ternary quantization with group-wise support
- **Hardware profiling** — CPU, GPU (NVIDIA, AMD, Apple Silicon), and system memory detection
- **Pipeline-based compiler** — Modular stages: Load → Analyze → Plan → Transform → Validate → Export
- **CLI interface** — Inspect, analyze, compile, and benchmark models from the command line

## Project Structure

```
src/ldmark/
├── __init__.py          # Package entry point
├── cli.py               # Command-line interface
├── analysis/            # Model analysis (tensors, parameters, architecture, storage, runtime)
├── compiler/            # Compilation pipeline (config, stages, pipeline orchestration)
├── compression/         # Quantization, binary/ternary, PrismML, experiments
├── file_io/             # Model loading, format detection, metadata
├── hardware/            # Hardware profiling and memory budgets
├── model_io/            # Bridge layer for model inspection
└── planning/            # Compression planning engine
```

## Installation

```bash
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Inspect a model file
ldmark inspect path/to/model.safetensors

# Analyze a model
ldmark analyze path/to/model.safetensors --format json

# Compile a model with compression
ldmark compile input_model.pt output_dir/ --precision int4 --strategy balanced

# Dry run (plan only, no execution)
ldmark compile input_model.pt output_dir/ --dry-run --verbose
```

### Python API

```python
from ldmark.analysis import ModelAnalyzer, load_numpy_dict
from ldmark.compression import quantize_groupwise, QuantizationConfig
from ldmark.hardware import detect_hardware
from ldmark.planning import CompressionPlanner

# Detect hardware
hw = detect_hardware()

# Analyze a model
tensors = load_numpy_dict("model.npz")
analyzer = ModelAnalyzer(model_id="my_model")
analysis = analyzer.analyze_numpy_tensors(tensors)

# Plan compression
planner = CompressionPlanner()
plan = planner.plan(analysis=analysis, hardware=hw, max_size_gb=4.0)
```

## Compression Methods

| Method | Bits/Weight | Description |
|--------|-------------|-------------|
| FP32 | 32.0 | Full precision baseline |
| FP16 | 16.0 | Half precision |
| INT8 | ~8.0 | 8-bit integer quantization |
| INT4 | ~4.0 | 4-bit integer quantization (group-wise) |
| Binary | ~1.0 | Binary quantization |
| Ternary | ~1.58 | Ternary quantization |
| PrismML Q1_0_g128 | ~1.0 | Custom 1-bit with group-128 scales |

## Pipeline Stages

1. **Load** — Load model from disk (auto-detect format)
2. **Analyze** — Extract tensor info, parameters, architecture
3. **Plan** — Generate compression plan based on constraints
4. **Transform** — Apply quantization/compression
5. **Validate** — Verify output quality (MAE, relative error)
6. **Export** — Save compiled model

## Testing

```bash
pytest tests/ -v
```

Test coverage includes:
- `tests/analysis/` — Tensor analysis, parameter counting, storage estimation
- `tests/compiler/` — Pipeline stages, config, CLI integration
- `tests/compression/` — Quantization correctness
- `tests/file_io/` — Model loading, format detection
- `tests/hardware/` — Hardware detection
- `tests/model_io/` — Bridge layer
- `tests/planning/` — Compression planning

## Experiments

Run compression experiments:

```bash
python experiments/compression/run_experiments.py
```

This runs a suite of quantization experiments (FP32/FP16 → INT8/INT4) and validates PrismML storage calculations.

## Requirements

- Python 3.10+
- NumPy
- PyTorch (optional, for `.pt`/`.pth` support)
- safetensors (optional, for `.safetensors` support)

## License

MIT
