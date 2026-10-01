"""
Tests for LDMARK hardware detection.

All tests are deterministic and do NOT depend on the actual developer machine.
Uses mocked hardware information exclusively.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import pytest
from unittest.mock import patch, MagicMock
from src.ldmark.hardware.profile import (
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
from src.ldmark.hardware.detection import DetectionResult, detect_hardware
from src.ldmark.hardware.memory_budget import MemoryBudget, MemoryBudgetResult, calculate_memory_budget


def make_cpu_profile(
    arch=CPUArchitecture.X86_64,
    vendor="GenuineIntel",
    model="Intel Core i7",
    cores=8,
    threads=16,
):
    return CPUProfile(
        architecture=arch,
        vendor=vendor,
        model=model,
        core_count=cores,
        thread_count=threads,
        simd=SIMDCapabilities(sse=True, sse2=True, avx=True, avx2=True),
    )


def make_gpu_nvidia():
    return GPUProfile(
        vendor=GPUVendor.NVIDIA,
        model="NVIDIA GeForce RTX 3080",
        vram_bytes=10 * 1024 ** 3,
        vram_gb=10.0,
        compute_capability="8.6",
        driver_version="535.104",
        runtime=GPURuntime.CUDA,
    )


def make_gpu_amd():
    return GPUProfile(
        vendor=GPUVendor.AMD,
        model="AMD Radeon RX 6800 XT",
        vram_bytes=16 * 1024 ** 3,
        vram_gb=16.0,
        runtime=GPURuntime.ROCM,
    )


def make_gpu_apple():
    return GPUProfile(
        vendor=GPUVendor.APPLE,
        model="Apple M2 Pro",
        runtime=GPURuntime.METAL,
        is_integrated=True,
    )


def make_system_profile(ram_gb=32, available_gb=16):
    return SystemProfile(
        total_ram_bytes=ram_gb * 1024 ** 3,
        total_ram_gb=float(ram_gb),
        available_ram_bytes=available_gb * 1024 ** 3,
        available_ram_gb=float(available_gb),
        operating_system="Linux",
        os_version="5.15",
        storage_bytes=512 * 1024 ** 3,
        storage_gb=512.0,
        python_version="3.11.0",
        platform_info="Linux x86_64",
    )


def make_hardware_profile(
    cpu=None, gpu=None, system=None, runtime=None,
):
    return HardwareProfile(
        cpu=cpu or make_cpu_profile(),
        gpu=gpu,
        system=system or make_system_profile(),
        runtime=runtime or RuntimeProfile(supported_backends=[]),
    )


class TestHardwareProfile:
    def test_cpu_only_profile(self):
        profile = make_hardware_profile(gpu=None)
        assert profile.cpu.architecture == CPUArchitecture.X86_64
        assert profile.cpu.core_count == 8
        assert profile.gpu is None
        assert not profile.has_gpu()

    def test_nvidia_gpu_profile(self):
        gpu = make_gpu_nvidia()
        profile = make_hardware_profile(gpu=gpu)
        assert profile.has_gpu()
        assert profile.gpu.vendor == GPUVendor.NVIDIA
        assert profile.gpu.vram_gb == 10.0
        assert profile.get_compute_backend() == GPURuntime.CUDA

    def test_amd_gpu_profile(self):
        gpu = make_gpu_amd()
        profile = make_hardware_profile(gpu=gpu)
        assert profile.has_gpu()
        assert profile.gpu.vendor == GPUVendor.AMD
        assert profile.get_compute_backend() == GPURuntime.ROCM

    def test_apple_gpu_profile(self):
        gpu = make_gpu_apple()
        profile = make_hardware_profile(gpu=gpu)
        assert profile.has_gpu()
        assert profile.gpu.vendor == GPUVendor.APPLE
        assert profile.gpu.is_integrated
        assert profile.get_compute_backend() == GPURuntime.METAL

    def test_unknown_hardware(self):
        profile = HardwareProfile(
            cpu=CPUProfile(architecture=CPUArchitecture.UNKNOWN),
            gpu=None,
            system=SystemProfile(),
            runtime=RuntimeProfile(),
        )
        assert not profile.has_gpu()
        assert profile.cpu.architecture == CPUArchitecture.UNKNOWN

    def test_to_dict_roundtrip(self):
        profile = make_hardware_profile(gpu=make_gpu_nvidia())
        d = profile.to_dict()
        assert d["cpu"]["architecture"] == "x86_64"
        assert d["gpu"]["vendor"] == "nvidia"
        assert d["system"]["total_ram_gb"] == 32.0

    def test_from_dict_roundtrip(self):
        profile = make_hardware_profile(gpu=make_gpu_nvidia())
        d = profile.to_dict()
        restored = HardwareProfile.from_dict(d)
        assert restored.cpu.architecture == profile.cpu.architecture
        assert restored.gpu.vendor == profile.gpu.vendor
        assert restored.system.total_ram_gb == profile.system.total_ram_gb


class TestHardwareDetection:
    def test_detect_hardware_returns_result(self):
        result = detect_hardware()
        assert isinstance(result, DetectionResult)
        assert isinstance(result.profile, HardwareProfile)
        assert result.timestamp != ""

    def test_detect_hardware_never_crashes(self):
        result = detect_hardware()
        assert result.profile is not None
        assert result.profile.cpu is not None
        assert result.profile.system is not None

    def test_detect_hardware_gpu_may_be_none(self):
        result = detect_hardware()
        if result.profile.gpu is None:
            assert any("No GPU" in w for w in result.warnings)

    def test_detect_hardware_to_dict(self):
        result = detect_hardware()
        d = result.to_dict()
        assert "profile" in d
        assert "success" in d
        assert "errors" in d
        assert "warnings" in d


class TestMemoryBudget:
    def test_memory_budget_creation(self):
        budget = MemoryBudget(
            max_model_storage_bytes=8 * 1024 ** 3,
            max_runtime_memory_bytes=16 * 1024 ** 3,
            minimum_free_memory_bytes=2 * 1024 ** 3,
        )
        assert budget.max_model_storage_bytes == 8 * 1024 ** 3
        assert budget.max_runtime_memory_bytes == 16 * 1024 ** 3

    def test_memory_budget_to_dict(self):
        budget = MemoryBudget(max_model_storage_bytes=1000)
        d = budget.to_dict()
        assert d["max_model_storage_bytes"] == 1000

    def test_memory_budget_from_dict(self):
        d = {"max_model_storage_bytes": 5000, "max_runtime_memory_bytes": 10000}
        budget = MemoryBudget.from_dict(d)
        assert budget.max_model_storage_bytes == 5000
        assert budget.max_runtime_memory_bytes == 10000

    def test_calculate_budget_no_constraints(self):
        result = calculate_memory_budget(
            model_storage_bytes=1000,
            weights_bytes=1000,
            kv_cache_bytes=500,
            activations_bytes=200,
            overhead_bytes=170,
        )
        assert result.fits_storage_budget
        assert result.fits_runtime_budget
        assert result.runtime_memory_bytes == 1870

    def test_calculate_budget_exceeds_storage(self):
        budget = MemoryBudget(max_model_storage_bytes=500)
        result = calculate_memory_budget(
            model_storage_bytes=1000,
            weights_bytes=1000,
            budget=budget,
        )
        assert not result.fits_storage_budget
        assert len(result.warnings) > 0

    def test_calculate_budget_exceeds_runtime(self):
        budget = MemoryBudget(max_runtime_memory_bytes=500)
        result = calculate_memory_budget(
            model_storage_bytes=100,
            weights_bytes=1000,
            budget=budget,
        )
        assert not result.fits_runtime_budget

    def test_calculate_budget_within_limits(self):
        budget = MemoryBudget(
            max_model_storage_bytes=10000,
            max_runtime_memory_bytes=20000,
        )
        result = calculate_memory_budget(
            model_storage_bytes=5000,
            weights_bytes=5000,
            kv_cache_bytes=1000,
            activations_bytes=500,
            overhead_bytes=650,
            budget=budget,
        )
        assert result.fits_storage_budget
        assert result.fits_runtime_budget

    def test_calculate_budget_pressure_ratio(self):
        result = calculate_memory_budget(
            model_storage_bytes=5000,
            weights_bytes=5000,
            available_memory_bytes=10000,
        )
        assert result.memory_pressure_ratio == 0.5

    def test_calculate_budget_minimum_free_memory(self):
        budget = MemoryBudget(minimum_free_memory_bytes=5000)
        result = calculate_memory_budget(
            model_storage_bytes=1000,
            weights_bytes=8000,
            budget=budget,
            available_memory_bytes=10000,
        )
        assert any("minimum" in w.lower() or "free" in w.lower() for w in result.warnings)

    def test_calculate_budget_zero_available(self):
        result = calculate_memory_budget(
            model_storage_bytes=1000,
            weights_bytes=1000,
            available_memory_bytes=0,
        )
        assert result.memory_pressure_ratio == 0.0

    def test_budget_result_to_dict(self):
        result = calculate_memory_budget(
            model_storage_bytes=1000,
            weights_bytes=1000,
        )
        d = result.to_dict()
        assert "model_storage_bytes" in d
        assert "runtime_memory_bytes" in d
        assert "fits_storage_budget" in d
        assert "fits_runtime_budget" in d
        assert "memory_pressure_ratio" in d
        assert "warnings" in d
        assert "assumptions" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])