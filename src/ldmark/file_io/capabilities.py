from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def is_torch_available() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_safetensors_available() -> bool:
    try:
        importlib.import_module("safetensors")
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def is_numpy_available() -> bool:
    try:
        importlib.import_module("numpy")
        return True
    except ImportError:
        return False


def get_torch_version() -> Optional[str]:
    if not is_torch_available():
        return None
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def get_safetensors_version() -> Optional[str]:
    if not is_safetensors_available():
        return None
    try:
        import safetensors
        return safetensors.__version__
    except Exception:
        return None


def get_capabilities() -> dict:
    return {
        "torch": {
            "available": is_torch_available(),
            "version": get_torch_version(),
        },
        "safetensors": {
            "available": is_safetensors_available(),
            "version": get_safetensors_version(),
        },
        "numpy": {
            "available": is_numpy_available(),
        },
    }