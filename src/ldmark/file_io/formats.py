from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union
import json


class ModelFormat(Enum):
    NUMPY = "numpy"
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    JSON_CONFIG = "json_config"
    SYNTHETIC = "synthetic"
    UNKNOWN = "unknown"


class ModelSource:
    def __init__(self, path: Union[str, Path], format: ModelFormat, is_directory: bool = False):
        self.path = Path(path)
        self.format = format
        self.is_directory = is_directory

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    def __repr__(self) -> str:
        return f"ModelSource(path={self.path}, format={self.format.value})"


def detect_format_from_path(path: Union[str, Path]) -> ModelSource:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_dir():
        return _detect_format_from_directory(p)

    return _detect_format_from_file(p)


def _detect_format_from_file(path: Path) -> ModelSource:
    suffix = path.suffix.lower()

    if suffix == ".npy" or suffix == ".npz":
        return ModelSource(path, ModelFormat.NUMPY)

    if suffix in (".pt", ".pth", ".bin"):
        return ModelSource(path, ModelFormat.PYTORCH)

    if suffix == ".safetensors":
        return ModelSource(path, ModelFormat.SAFETENSORS)

    if suffix == ".json":
        return ModelSource(path, ModelFormat.JSON_CONFIG)

    return ModelSource(path, ModelFormat.UNKNOWN)


def _detect_format_from_directory(path: Path) -> ModelSource:
    files = list(path.iterdir())
    file_names = {f.name.lower() for f in files}

    # Single priority chain: safetensors > pytorch > numpy > json-config-only.
    # Config weights (architectures/model_type) only decide when no tensor files
    # are present, so a config.json next to NumPy weights resolves to NUMPY and
    # the NUMPY loader picks up the config as a sidecar.
    has_config_weights = False
    if "config.json" in file_names:
        try:
            with open(path / "config.json", "r") as f:
                config = json.load(f)
            has_config_weights = "architectures" in config or "model_type" in config
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if any(f.suffix == ".safetensors" for f in files):
        return ModelSource(path, ModelFormat.SAFETENSORS, is_directory=True)

    if any(f.suffix in (".bin", ".pt", ".pth") for f in files):
        return ModelSource(path, ModelFormat.PYTORCH, is_directory=True)

    if any(f.suffix in (".npy", ".npz") for f in files):
        return ModelSource(path, ModelFormat.NUMPY, is_directory=True)

    if has_config_weights or any(f.suffix == ".json" for f in files):
        return ModelSource(path, ModelFormat.JSON_CONFIG, is_directory=True)

    return ModelSource(path, ModelFormat.UNKNOWN, is_directory=True)


def detect_format_from_content(path: Union[str, Path]) -> ModelSource:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    if p.is_dir():
        return _detect_format_from_directory(p)

    suffix = p.suffix.lower()

    if suffix == ".json":
        try:
            with open(p, "r") as f:
                json.load(f)
            return ModelSource(p, ModelFormat.JSON_CONFIG)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ModelSource(p, ModelFormat.UNKNOWN)

    if suffix in (".npy", ".npz"):
        return ModelSource(p, ModelFormat.NUMPY)

    if suffix == ".safetensors":
        return ModelSource(p, ModelFormat.SAFETENSORS)

    if suffix in (".pt", ".pth", ".bin"):
        return ModelSource(p, ModelFormat.PYTORCH)

    return ModelSource(p, ModelFormat.UNKNOWN)