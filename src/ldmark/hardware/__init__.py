"""
LDMARK Hardware Profiling System

Structured representation of target hardware for compression planning.
Supports CPU, GPU (NVIDIA, AMD, Apple Silicon), system memory, and runtime backends.
"""

from .profile import (
    CPUArchitecture,
    GPUVendor,
    GPURuntime,
    SIMDCapabilities,
    CPUProfile,
    GPUProfile,
    SystemProfile,
    RuntimeProfile,
    HardwareProfile,
)
from .detection import detect_hardware, DetectionResult
from .memory_budget import MemoryBudget, MemoryBudgetResult

__all__ = [
    "CPUArchitecture",
    "GPUVendor",
    "GPURuntime",
    "SIMDCapabilities",
    "CPUProfile",
    "GPUProfile",
    "SystemProfile",
    "RuntimeProfile",
    "HardwareProfile",
    "detect_hardware",
    "DetectionResult",
    "MemoryBudget",
    "MemoryBudgetResult",
]