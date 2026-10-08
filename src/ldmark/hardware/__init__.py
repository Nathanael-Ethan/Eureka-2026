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
    apple_silicon_cpu_profile,
    laptop_cpu_profile_4gb,
    laptop_cpu_profile_8gb,
)
from .detection import detect_hardware, DetectionResult
from .memory_budget import (
    MemoryBudget,
    MemoryBudgetResult,
    laptop_budget_4gb,
    laptop_budget_8gb,
    check_runtime_against_budget,
)

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
    "apple_silicon_cpu_profile",
    "laptop_cpu_profile_4gb",
    "laptop_cpu_profile_8gb",
    "laptop_budget_4gb",
    "laptop_budget_8gb",
    "check_runtime_against_budget",
]