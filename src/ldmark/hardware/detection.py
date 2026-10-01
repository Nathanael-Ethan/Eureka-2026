"""
Safe hardware detection for LDMARK.

Collects only information that can be reliably detected.
Optional dependencies remain optional.
If GPU information cannot be detected, gpu = None.
Does not fabricate hardware information.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .profile import (
    CPUArchitecture,
    CPUProfile,
    GPUProfile,
    GPUVendor,
    GPURuntime,
    HardwareProfile,
    RuntimeProfile,
    SIMDCapabilities,
    SystemProfile,
)


@dataclass(frozen=True)
class DetectionResult:
    """Result of hardware detection with metadata."""
    profile: HardwareProfile
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }


def _detect_cpu_architecture() -> CPUArchitecture:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return CPUArchitecture.X86_64
    if machine in ("arm64", "aarch64"):
        return CPUArchitecture.ARM64
    if machine.startswith("arm"):
        return CPUArchitecture.ARM
    return CPUArchitecture.UNKNOWN


def _detect_cpu_simd() -> SIMDCapabilities:
    simd = SIMDCapabilities()
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().lower()
            simd = SIMDCapabilities(
                sse="sse" in cpuinfo,
                sse2="sse2" in cpuinfo,
                sse3="sse3" in cpuinfo or "pni" in cpuinfo,
                ssse3="ssse3" in cpuinfo,
                sse4_1="sse4_1" in cpuinfo,
                sse4_2="sse4_2" in cpuinfo,
                avx="avx" in cpuinfo,
                avx2="avx2" in cpuinfo,
                avx512="avx512" in cpuinfo,
                neon="neon" in cpuinfo,
                sve="sve" in cpuinfo,
            )
        elif platform.system() == "Darwin":
            simd = SIMDCapabilities(neon=True)
    except Exception:
        pass
    return simd


def _detect_cpu() -> CPUProfile:
    arch = _detect_cpu_architecture()
    simd = _detect_cpu_simd()

    core_count = os.cpu_count() or 0
    thread_count = core_count

    vendor = "unknown"
    model = "unknown"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "vendor_id" in line.lower():
                        vendor = line.split(":")[-1].strip()
                    elif "model name" in line.lower():
                        model = line.split(":")[-1].strip()
                    if vendor != "unknown" and model != "unknown":
                        break
        elif platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.vendor"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    vendor = result.stdout.strip()
            except Exception:
                pass
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    model = result.stdout.strip()
            except Exception:
                pass
        elif platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                    if len(lines) > 1:
                        model = lines[1]
            except Exception:
                pass
    except Exception:
        pass

    return CPUProfile(
        architecture=arch,
        vendor=vendor,
        model=model,
        core_count=core_count,
        thread_count=thread_count,
        simd=simd,
    )


def _detect_gpu_nvidia() -> Optional[GPUProfile]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return None
        name, mem_str, driver, compute_cap = parts[0], parts[1], parts[2], parts[3]
        vram_bytes = 0
        try:
            mem_val = int(mem_str.split()[0])
            if "MiB" in mem_str:
                vram_bytes = mem_val * 1024 * 1024
            elif "GiB" in mem_str:
                vram_bytes = mem_val * 1024 * 1024 * 1024
        except (ValueError, IndexError):
            pass
        return GPUProfile(
            vendor=GPUVendor.NVIDIA,
            model=name,
            vram_bytes=vram_bytes if vram_bytes > 0 else None,
            vram_gb=round(vram_bytes / (1024 ** 3), 2) if vram_bytes > 0 else None,
            compute_capability=compute_cap,
            driver_version=driver,
            runtime=GPURuntime.CUDA,
        )
    except Exception:
        return None


def _detect_gpu_amd() -> Optional[GPUProfile]:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        return GPUProfile(
            vendor=GPUVendor.AMD,
            model="AMD GPU (ROCm)",
            runtime=GPURuntime.ROCM,
        )
    except Exception:
        return None


def _detect_gpu_apple() -> Optional[GPUProfile]:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        output = result.stdout
        if "Apple" not in output and "M1" not in output and "M2" not in output and "M3" not in output:
            return None
        model = "Apple Silicon"
        for line in output.split("\n"):
            if "Chipset Model:" in line:
                model = line.split(":")[-1].strip()
                break
        return GPUProfile(
            vendor=GPUVendor.APPLE,
            model=model,
            runtime=GPURuntime.METAL,
            is_integrated=True,
        )
    except Exception:
        return None


def _detect_gpu() -> Optional[GPUProfile]:
    gpu = _detect_gpu_nvidia()
    if gpu:
        return gpu
    gpu = _detect_gpu_amd()
    if gpu:
        return gpu
    gpu = _detect_gpu_apple()
    if gpu:
        return gpu
    return None


def _detect_system() -> SystemProfile:
    total_ram = 0
    available_ram = None
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal:" in line:
                        total_ram = int(line.split()[1]) * 1024
                    elif "MemAvailable:" in line:
                        available_ram = int(line.split()[1]) * 1024
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                total_ram = int(result.stdout.strip())
        elif platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
                if len(lines) > 1:
                    total_ram = int(lines[1])
    except Exception:
        pass

    storage_bytes = None
    try:
        if hasattr(os, "statvfs"):
            stat = os.statvfs(".")
            storage_bytes = stat.f_blocks * stat.f_frsize
    except Exception:
        pass

    return SystemProfile(
        total_ram_bytes=total_ram,
        total_ram_gb=round(total_ram / (1024 ** 3), 2) if total_ram > 0 else 0.0,
        available_ram_bytes=available_ram,
        available_ram_gb=round(available_ram / (1024 ** 3), 2) if available_ram else None,
        operating_system=platform.system(),
        os_version=platform.version(),
        storage_bytes=storage_bytes,
        storage_gb=round(storage_bytes / (1024 ** 3), 2) if storage_bytes else None,
        python_version=platform.python_version(),
        platform_info=platform.platform(),
    )


def _detect_runtime(gpu: Optional[GPUProfile]) -> RuntimeProfile:
    backends = []
    cuda_version = None
    rocm_version = None
    metal_available = False
    opencl_available = False
    vulkan_available = False

    if gpu:
        if gpu.runtime == GPURuntime.CUDA:
            backends.append(GPURuntime.CUDA)
            try:
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "release" in line.lower():
                            cuda_version = line.split("release")[-1].strip().rstrip(",")
                            break
            except Exception:
                pass
        elif gpu.runtime == GPURuntime.ROCM:
            backends.append(GPURuntime.ROCM)
        elif gpu.runtime == GPURuntime.METAL:
            backends.append(GPURuntime.METAL)
            metal_available = True

    try:
        result = subprocess.run(
            ["clinfo"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "platform name" in result.stdout.lower():
            opencl_available = True
            if GPURuntime.OPENCL not in backends:
                backends.append(GPURuntime.OPENCL)
    except Exception:
        pass

    return RuntimeProfile(
        supported_backends=backends,
        cuda_version=cuda_version,
        rocm_version=rocm_version,
        metal_available=metal_available,
        opencl_available=opencl_available,
        vulkan_available=vulkan_available,
    )


def detect_hardware() -> DetectionResult:
    """
    Safely detect hardware information.

    Returns a DetectionResult with a HardwareProfile.
    Never raises exceptions - all errors are captured in the result.
    """
    errors = []
    warnings = []
    timestamp = datetime.now().isoformat()

    try:
        cpu = _detect_cpu()
    except Exception as e:
        errors.append(f"CPU detection failed: {e}")
        cpu = CPUProfile()

    try:
        gpu = _detect_gpu()
    except Exception as e:
        errors.append(f"GPU detection failed: {e}")
        gpu = None

    if gpu is None:
        warnings.append("No GPU detected. GPU acceleration will not be available.")

    try:
        system = _detect_system()
    except Exception as e:
        errors.append(f"System detection failed: {e}")
        system = SystemProfile()

    try:
        runtime = _detect_runtime(gpu)
    except Exception as e:
        errors.append(f"Runtime detection failed: {e}")
        runtime = RuntimeProfile()

    profile = HardwareProfile(
        cpu=cpu,
        gpu=gpu,
        system=system,
        runtime=runtime,
        detection_timestamp=timestamp,
        detection_notes=warnings + errors,
    )

    return DetectionResult(
        profile=profile,
        success=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        timestamp=timestamp,
    )