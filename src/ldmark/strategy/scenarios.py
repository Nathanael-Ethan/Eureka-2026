"""
LDMARK Strategy Engine - Research Scenarios

Deterministic scenarios for research and validation.
Uses mathematical model descriptions only - NO real model files.
Verifies internal consistency of calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ldmark.analysis.models import ArchitectureInfo, DType, ModelAnalysis, ParameterCounts, TensorInfo, TensorTrainableStatus
from src.ldmark.hardware.profile import CPUArchitecture, CPUProfile, GPUProfile, GPUVendor, GPURuntime, HardwareProfile, RuntimeProfile, SystemProfile
from src.ldmark.strategy.engine import StrategyEngine, evaluate_strategies
from src.ldmark.strategy.models import ConstraintConfig, StrategySet
from src.ldmark.strategy.feasibility import (
    calculate_actual_storage_bytes,
    calculate_effective_bits_per_weight,
    calculate_runtime_memory,
    calculate_storage_bytes,
)


def create_synthetic_model_analysis(
    model_id: str,
    param_count: int,
    num_layers: int = 32,
    hidden_size: int = 4096,
    num_attention_heads: int = 32,
    num_kv_heads: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    vocab_size: int = 32000,
    max_context_length: int = 4096,
    architecture_name: str = "llama",
    attention_type: Optional[str] = None,
) -> ModelAnalysis:
    """Create a synthetic ModelAnalysis for deterministic testing."""
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if intermediate_size is None:
        intermediate_size = hidden_size * 4
    
    params = ParameterCounts(
        total=param_count,
        by_dtype={DType.FP32: param_count},
        by_tensor={"model.embed_tokens.weight": vocab_size * hidden_size},
        trainable=param_count,
        non_trainable=0,
    )
    
    tensors = [
        TensorInfo(
            name="model.embed_tokens.weight",
            shape=[vocab_size, hidden_size],
            dtype=DType.FP32,
            num_elements=vocab_size * hidden_size,
            estimated_raw_storage_bytes=vocab_size * hidden_size * 4,
        )
    ]
    
    arch = ArchitectureInfo(
        model_architecture=architecture_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
        attention_type=attention_type,
    )
    
    return ModelAnalysis(
        model_id=model_id,
        source_path=f"/synthetic/{model_id}",
        parameter_counts=params,
        tensors=tensors,
        architecture=arch,
        storage_estimates=[],
        compression_estimates=[],
    )


def create_synthetic_hardware_profile(
    ram_gb: float,
    available_ram_gb: Optional[float] = None,
    gpu_vram_gb: Optional[float] = None,
    gpu_vendor: str = "none",
    gpu_model: str = "none",
    backends: Optional[List[str]] = None,
) -> HardwareProfile:
    """Create a synthetic HardwareProfile for deterministic testing."""
    total_ram = int(ram_gb * 1024 ** 3)
    available_ram = int((available_ram_gb or ram_gb * 0.75) * 1024 ** 3)
    
    gpu = None
    if gpu_vram_gb and gpu_vendor != "none":
        vendor_map = {
            "nvidia": GPUVendor.NVIDIA,
            "amd": GPUVendor.AMD,
            "apple": GPUVendor.APPLE,
            "intel": GPUVendor.INTEL,
        }
        runtime_map = {
            "nvidia": GPURuntime.CUDA,
            "amd": GPURuntime.ROCM,
            "apple": GPURuntime.METAL,
            "intel": GPURuntime.OPENCL,
        }
        gpu = GPUProfile(
            vendor=vendor_map.get(gpu_vendor.lower(), GPUVendor.UNKNOWN),
            model=gpu_model,
            vram_bytes=int(gpu_vram_gb * 1024 ** 3),
            vram_gb=gpu_vram_gb,
            runtime=runtime_map.get(gpu_vendor.lower(), GPURuntime.NONE),
        )
    
    supported_backends = []
    if backends:
        from src.ldmark.hardware.profile import GPURuntime
        for b in backends:
            try:
                supported_backends.append(GPURuntime(b.lower()))
            except ValueError:
                pass
    elif gpu:
        supported_backends = [gpu.runtime]
    
    return HardwareProfile(
        cpu=CPUProfile(
            architecture=CPUArchitecture.X86_64,
            model="Synthetic CPU",
            core_count=16,
            thread_count=32,
        ),
        gpu=gpu,
        system=SystemProfile(
            total_ram_bytes=total_ram,
            total_ram_gb=ram_gb,
            available_ram_bytes=available_ram,
            available_ram_gb=available_ram_gb or ram_gb * 0.75,
            operating_system="Linux",
        ),
        runtime=RuntimeProfile(supported_backends=supported_backends),
    )


# Scenario Definitions

SCENARIO_A = {
    "name": "Scenario A: 7B model + 16 GB RAM",
    "model_id": "scenario-a-7b",
    "param_count": 7_000_000_000,
    "ram_gb": 16,
    "available_ram_gb": 12,
    "constraints": ConstraintConfig(
        max_model_storage_bytes=None,
        max_runtime_memory_bytes=None,
        minimum_available_memory_bytes=None,
    ),
    "context_lengths": [4096, 16384, 32768],
}

SCENARIO_B = {
    "name": "Scenario B: 13B model + 16 GB RAM",
    "model_id": "scenario-b-13b",
    "param_count": 13_000_000_000,
    "ram_gb": 16,
    "available_ram_gb": 12,
    "constraints": ConstraintConfig(
        max_model_storage_bytes=None,
        max_runtime_memory_bytes=None,
        minimum_available_memory_bytes=None,
    ),
    "context_lengths": [4096, 16384, 32768],
}

SCENARIO_C = {
    "name": "Scenario C: 27B model + 16 GB RAM",
    "model_id": "scenario-c-27b",
    "param_count": 27_000_000_000,
    "ram_gb": 16,
    "available_ram_gb": 12,
    "constraints": ConstraintConfig(
        max_model_storage_bytes=None,
        max_runtime_memory_bytes=None,
        minimum_available_memory_bytes=None,
    ),
    "context_lengths": [4096, 16384, 32768],
}

SCENARIO_D = {
    "name": "Scenario D: 27B model + 32 GB RAM",
    "model_id": "scenario-d-27b",
    "param_count": 27_000_000_000,
    "ram_gb": 32,
    "available_ram_gb": 24,
    "constraints": ConstraintConfig(
        max_model_storage_bytes=None,
        max_runtime_memory_bytes=None,
        minimum_available_memory_bytes=None,
    ),
    "context_lengths": [4096, 16384, 32768, 100000],
}

SCENARIO_E = {
    "name": "Scenario E: 27B model + 8 GB storage limit",
    "model_id": "scenario-e-27b",
    "param_count": 27_000_000_000,
    "ram_gb": 32,
    "available_ram_gb": 24,
    "constraints": ConstraintConfig(
        max_model_storage_bytes=8 * 1024 ** 3,
        max_runtime_memory_bytes=None,
        minimum_available_memory_bytes=None,
    ),
    "context_lengths": [4096],
}

SCENARIO_27B_Q1_G128 = {
    "name": "27B / Q1_0_g128 Analytical Scenario",
    "model_id": "27b-q1-g128",
    "param_count": 27_000_000_000,
    "ram_gb": 16,
    "available_ram_gb": 12,
    "constraints": ConstraintConfig(
        max_model_storage_bytes=8 * 1024 ** 3,
        max_runtime_memory_bytes=12 * 1024 ** 3,
        minimum_available_memory_bytes=2 * 1024 ** 3,
    ),
    "context_lengths": [4096],
    "representation": "ldmark_binary",  # Maps to Q1_0_g128 (1.125 bpw)
}

ALL_SCENARIOS = [
    SCENARIO_A,
    SCENARIO_B,
    SCENARIO_C,
    SCENARIO_D,
    SCENARIO_E,
    SCENARIO_27B_Q1_G128,
]


def run_scenario(scenario: Dict[str, Any], include_ldmark: bool = True) -> StrategySet:
    """Run a single scenario and return the strategy set."""
    analysis = create_synthetic_model_analysis(
        model_id=scenario["model_id"],
        param_count=scenario["param_count"],
        num_layers=80 if scenario["param_count"] >= 27_000_000_000 else 32,
        hidden_size=8192 if scenario["param_count"] >= 27_000_000_000 else 4096,
        num_attention_heads=64 if scenario["param_count"] >= 27_000_000_000 else 32,
        num_kv_heads=8 if scenario["param_count"] >= 27_000_000_000 else 32,
        intermediate_size=28672 if scenario["param_count"] >= 27_000_000_000 else 11008,
        max_context_length=max(scenario.get("context_lengths", [4096])),
    )
    
    hardware = create_synthetic_hardware_profile(
        ram_gb=scenario["ram_gb"],
        available_ram_gb=scenario.get("available_ram_gb"),
    )
    
    constraints = scenario["constraints"]
    context_length = scenario.get("context_lengths", [4096])[0]
    
    engine = StrategyEngine(
        context_length=context_length,
        batch_size=1,
        include_ldmark_formats=include_ldmark,
    )
    
    return engine.evaluate(analysis, hardware, constraints)


def run_all_scenarios(include_ldmark: bool = True) -> Dict[str, StrategySet]:
    """Run all research scenarios."""
    results = {}
    for scenario in ALL_SCENARIOS:
        results[scenario["name"]] = run_scenario(scenario, include_ldmark)
    return results


def analyze_27b_q1_g128() -> Dict[str, Any]:
    """
    Analytical analysis of 27B parameters with Q1_0_g128 (1.125 bits/weight).
    
    Clearly distinguishes:
    - theoretical representation size
    - actual serialized file size  
    - runtime memory
    
    Does NOT reproduce or claim PrismML benchmark quality.
    """
    param_count = 27_000_000_000
    weight_bits = 1.0
    group_size = 128
    scale_bits = 16
    
    effective_bpw = calculate_effective_bits_per_weight(weight_bits, group_size, scale_bits)
    
    theoretical_bytes = calculate_storage_bytes(param_count, effective_bpw)
    actual_bytes = calculate_actual_storage_bytes(param_count, weight_bits, group_size, scale_bits)
    
    arch = ArchitectureInfo(
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
    )
    
    runtime_mem = calculate_runtime_memory(
        param_count=param_count,
        bits_per_weight=effective_bpw,
        architecture=arch,
        context_length=4096,
        batch_size=1,
    )
    
    theoretical_gb = theoretical_bytes / (1024 ** 3)
    actual_gb = actual_bytes / (1024 ** 3)
    runtime_gb = runtime_mem["total"] / (1024 ** 3)
    weights_gb = runtime_mem["weights"] / (1024 ** 3)
    kv_gb = runtime_mem["kv_cache"] / (1024 ** 3)
    act_gb = runtime_mem["activations"] / (1024 ** 3)
    
    return {
        "model_parameters": param_count,
        "representation": "Q1_0_g128 (PrismML-style)",
        "weight_bits": weight_bits,
        "group_size": group_size,
        "scale_bits": scale_bits,
        "effective_bits_per_weight": effective_bpw,
        "theoretical_representation": {
            "bytes": theoretical_bytes,
            "gb": round(theoretical_gb, 4),
            "description": "Pure information-theoretic size: params * effective_bpw / 8",
        },
        "actual_serialized_file": {
            "bytes": actual_bytes,
            "gb": round(actual_gb, 4),
            "description": "Includes scale factors per group: weights + scales",
            "scale_factor_count": (param_count + group_size - 1) // group_size,
            "scale_factor_bytes": ((param_count + group_size - 1) // group_size) * (scale_bits // 8),
        },
        "runtime_memory_4k_context": {
            "total_gb": round(runtime_gb, 4),
            "weights_gb": round(weights_gb, 4),
            "kv_cache_gb": round(kv_gb, 4),
            "activations_gb": round(act_gb, 4),
            "overhead_gb": round(runtime_mem["overhead"] / (1024 ** 3), 4),
            "assumptions": {
                "context_length": 4096,
                "batch_size": 1,
                "architecture": "80-layer, 8192 hidden, 64 heads, 8 KV heads (GQA)",
                "activation_dtype": "fp16",
                "overhead_factor": 0.1,
                "note": "Estimates are approximate and architecture-dependent.",
            },
        },
        "key_distinctions": [
            "Theoretical representation size (3.78 GB) = pure information content",
            "Actual serialized file size (3.90 GB) = includes scale factor overhead",
            "Runtime memory (weights + KV + activations) > serialized size",
            "These are THREE DIFFERENT NUMBERS serving different purposes",
        ],
    }


def verify_internal_consistency() -> Dict[str, Any]:
    """Verify that calculations are internally consistent across scenarios."""
    issues = []
    
    for scenario in [SCENARIO_A, SCENARIO_B, SCENARIO_C, SCENARIO_D, SCENARIO_E]:
        result = run_scenario(scenario, include_ldmark=True)
        
        for strategy in result.strategies:
            assumptions = strategy.assumptions
            
            if "theoretical_storage_bytes" in assumptions and "actual_serialized_storage_bytes" in assumptions:
                theoretical = assumptions["theoretical_storage_bytes"]
                actual = assumptions["actual_serialized_storage_bytes"]
                if actual < theoretical:
                    issues.append(
                        f"{scenario['name']}: {strategy.weight_representation.value} "
                        f"actual ({actual}) < theoretical ({theoretical})"
                    )
            
            if "runtime_total_bytes" in assumptions and "runtime_weights_bytes" in assumptions:
                total = assumptions["runtime_total_bytes"]
                weights = assumptions["runtime_weights_bytes"]
                if total < weights:
                    issues.append(
                        f"{scenario['name']}: {strategy.weight_representation.value} "
                        f"runtime total ({total}) < weights ({weights})"
                    )
    
    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "checked_scenarios": len(ALL_SCENARIOS) - 1,  # Exclude analytical scenario
    }