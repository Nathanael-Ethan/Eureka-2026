"""
LDMARK Hardware Profile System

Structured representation of target hardware for compression planning.
Supports CPU, GPU (NVIDIA, AMD, Apple Silicon), system memory, and runtime backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import platform
import sys


class CPUArchitecture(Enum):
    """Known CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM = "arm"
    UNKNOWN = "unknown"


class GPUVendor(Enum):
    """Known GPU vendors."""
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    QUALCOMM = "qualcomm"
    UNKNOWN = "unknown"


class GPURuntime(Enum):
    """Known GPU compute runtimes/backends."""
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    OPENCL = "opencl"
    VULKAN = "vulkan"
    NONE = "none"


@dataclass(frozen=True)
class SIMDCapabilities:
    """CPU SIMD instruction set capabilities."""
    sse: bool = False
    sse2: bool = False
    sse3: bool = False
    ssse3: bool = False
    sse4_1: bool = False
    sse4_2: bool = False
    avx: bool = False
    avx2: bool = False
    avx512: bool = False
    neon: bool = False
    sve: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {k: v for k, v in self.__dict__.items() if v}

    @classmethod
    def from_dict(cls, data: Dict[str, bool]) -> SIMDCapabilities:
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass(frozen=True)
class CPUProfile:
    """CPU hardware profile."""
    architecture: CPUArchitecture = CPUArchitecture.UNKNOWN
    vendor: str = "unknown"
    model: str = "unknown"
    core_count: int = 0
    thread_count: int = 0
    base_frequency_ghz: Optional[float] = None
    max_frequency_ghz: Optional[float] = None
    cache_l1_kb: Optional[int] = None
    cache_l2_kb: Optional[int] = None
    cache_l3_kb: Optional[int] = None
    simd: SIMDCapabilities = field(default_factory=SIMDCapabilities)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "vendor": self.vendor,
            "model": self.model,
            "core_count": self.core_count,
            "thread_count": self.thread_count,
            "base_frequency_ghz": self.base_frequency_ghz,
            "max_frequency_ghz": self.max_frequency_ghz,
            "cache_l1_kb": self.cache_l1_kb,
            "cache_l2_kb": self.cache_l2_kb,
            "cache_l3_kb": self.cache_l3_kb,
            "simd": self.simd.to_dict(),
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CPUProfile:
        simd = SIMDCapabilities.from_dict(data.get("simd", {}))
        arch = CPUArchitecture(data.get("architecture", "unknown"))
        return cls(
            architecture=arch,
            vendor=data.get("vendor", "unknown"),
            model=data.get("model", "unknown"),
            core_count=data.get("core_count", 0),
            thread_count=data.get("thread_count", 0),
            base_frequency_ghz=data.get("base_frequency_ghz"),
            max_frequency_ghz=data.get("max_frequency_ghz"),
            cache_l1_kb=data.get("cache_l1_kb"),
            cache_l2_kb=data.get("cache_l2_kb"),
            cache_l3_kb=data.get("cache_l3_kb"),
            simd=simd,
            flags=data.get("flags", []),
        )


@dataclass(frozen=True)
class GPUProfile:
    """GPU hardware profile."""
    vendor: GPUVendor = GPUVendor.UNKNOWN
    model: str = "unknown"
    vram_bytes: Optional[int] = None
    vram_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    runtime: GPURuntime = GPURuntime.NONE
    backend_info: Dict[str, Any] = field(default_factory=dict)
    is_integrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor.value,
            "model": self.model,
            "vram_bytes": self.vram_bytes,
            "vram_gb": self.vram_gb,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "runtime": self.runtime.value,
            "backend_info": self.backend_info,
            "is_integrated": self.is_integrated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GPUProfile:
        vendor = GPUVendor(data.get("vendor", "unknown"))
        runtime = GPURuntime(data.get("runtime", "none"))
        return cls(
            vendor=vendor,
            model=data.get("model", "unknown"),
            vram_bytes=data.get("vram_bytes"),
            vram_gb=data.get("vram_gb"),
            compute_capability=data.get("compute_capability"),
            driver_version=data.get("driver_version"),
            runtime=runtime,
            backend_info=data.get("backend_info", {}),
            is_integrated=data.get("is_integrated", False),
        )


@dataclass(frozen=True)
class SystemProfile:
    """System-level hardware profile."""
    total_ram_bytes: int = 0
    total_ram_gb: float = 0.0
    available_ram_bytes: Optional[int] = None
    available_ram_gb: Optional[float] = None
    operating_system: str = "unknown"
    os_version: str = "unknown"
    storage_bytes: Optional[int] = None
    storage_gb: Optional[float] = None
    python_version: str = ""
    platform_info: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_bytes": self.total_ram_bytes,
            "total_ram_gb": self.total_ram_gb,
            "available_ram_bytes": self.available_ram_bytes,
            "available_ram_gb": self.available_ram_gb,
            "operating_system": self.operating_system,
            "os_version": self.os_version,
            "storage_bytes": self.storage_bytes,
            "storage_gb": self.storage_gb,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SystemProfile:
        return cls(
            total_ram_bytes=data.get("total_ram_bytes", 0),
            total_ram_gb=data.get("total_ram_gb", 0.0),
            available_ram_bytes=data.get("available_ram_bytes"),
            available_ram_gb=data.get("available_ram_gb"),
            operating_system=data.get("operating_system", "unknown"),
            os_version=data.get("os_version", "unknown"),
            storage_bytes=data.get("storage_bytes"),
            storage_gb=data.get("storage_gb"),
            python_version=data.get("python_version", ""),
            platform_info=data.get("platform_info", ""),
        )


@dataclass(frozen=True)
class RuntimeProfile:
    """Runtime/software environment profile."""
    supported_backends: List[GPURuntime] = field(default_factory=list)
    available_memory_bytes: Optional[int] = None
    available_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    rocm_version: Optional[str] = None
    metal_available: bool = False
    opencl_available: bool = False
    vulkan_available: bool = False
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported_backends": [b.value for b in self.supported_backends],
            "available_memory_bytes": self.available_memory_bytes,
            "available_memory_gb": self.available_memory_gb,
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
            "metal_available": self.metal_available,
            "opencl_available": self.opencl_available,
            "vulkan_available": self.vulkan_available,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RuntimeProfile:
        backends = [GPURuntime(b) for b in data.get("supported_backends", [])]
        return cls(
            supported_backends=backends,
            available_memory_bytes=data.get("available_memory_bytes"),
            available_memory_gb=data.get("available_memory_gb"),
            cuda_version=data.get("cuda_version"),
            rocm_version=data.get("rocm_version"),
            metal_available=data.get("metal_available", False),
            opencl_available=data.get("opencl_available", False),
            vulkan_available=data.get("vulkan_available", False),
            constraints=data.get("constraints", {}),
        )


@dataclass(frozen=True)
class HardwareProfile:
    """Complete hardware profile for a target system."""
    cpu: CPUProfile = field(default_factory=CPUProfile)
    gpu: Optional[GPUProfile] = None
    system: SystemProfile = field(default_factory=SystemProfile)
    runtime: RuntimeProfile = field(default_factory=RuntimeProfile)
    detection_timestamp: str = ""
    detection_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": self.cpu.to_dict(),
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "system": self.system.to_dict(),
            "runtime": self.runtime.to_dict(),
            "detection_timestamp": self.detection_timestamp,
            "detection_notes": self.detection_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HardwareProfile:
        cpu = CPUProfile.from_dict(data.get("cpu", {}))
        gpu_data = data.get("gpu")
        gpu = GPUProfile.from_dict(gpu_data) if gpu_data else None
        system = SystemProfile.from_dict(data.get("system", {}))
        runtime = RuntimeProfile.from_dict(data.get("runtime", {}))
        return cls(
            cpu=cpu,
            gpu=gpu,
            system=system,
            runtime=runtime,
            detection_timestamp=data.get("detection_timestamp", ""),
            detection_notes=data.get("detection_notes", []),
        )

    def has_gpu(self) -> bool:
        """Check if a GPU is detected."""
        return self.gpu is not None and self.gpu.vendor != GPUVendor.UNKNOWN

    def get_compute_backend(self) -> GPURuntime:
        """Get the primary compute backend."""
        if self.gpu and self.gpu.runtime != GPURuntime.NONE:
            return self.gpu.runtime
        if self.runtime.supported_backends:
            return self.runtime.supported_backends[0]
        return GPURuntime.NONE


# --- Predefined laptop profiles (M3: laptop-ready runtime) ---
# Storage size vs runtime memory are kept separate: these profiles describe
# RUNTIME memory available for weights + KV + activations + overhead.

def _laptop_cpu_profile(total_ram_bytes: int, model: str) -> HardwareProfile:
    import platform as _platform
    gb = round(total_ram_bytes / (1024 ** 3), 2)
    return HardwareProfile(
        cpu=CPUProfile(
            architecture=CPUArchitecture.ARM64 if _platform.machine().lower() in ("arm64", "aarch64") else CPUArchitecture.X86_64,
            vendor="Apple" if "apple" in model.lower() or "m1" in model.lower() else "unknown",
            model=model,
            core_count=8,
            thread_count=8,
            simd=SIMDCapabilities(neon=True),
        ),
        gpu=None,  # CPU-only laptop profile: no GPU execution assumed
        system=SystemProfile(
            total_ram_bytes=total_ram_bytes,
            total_ram_gb=gb,
            operating_system=_platform.system(),
            os_version=_platform.version(),
            python_version=_platform.python_version(),
            platform_info=_platform.platform(),
        ),
        runtime=RuntimeProfile(supported_backends=[], metal_available=False),
        detection_timestamp="",
        detection_notes=[f"Predefined laptop CPU profile ({gb}GB total RAM, CPU-only, Apple Silicon compatible)."],
    )


def apple_silicon_cpu_profile(total_ram_bytes: int = 8 * 1024 ** 3) -> HardwareProfile:
    """Apple Silicon laptop, CPU-only execution (no Metal kernels assumed)."""
    return _laptop_cpu_profile(total_ram_bytes, "Apple Silicon (M1 baseline, CPU)")


def laptop_cpu_profile_4gb() -> HardwareProfile:
    """4GB laptop-class CPU profile (tightest supported budget)."""
    return _laptop_cpu_profile(4 * 1024 ** 3, "Generic laptop CPU (4GB class)")


def laptop_cpu_profile_8gb() -> HardwareProfile:
    """8GB laptop-class CPU profile (standard laptop budget)."""
    return _laptop_cpu_profile(8 * 1024 ** 3, "Generic laptop CPU (8GB class)")