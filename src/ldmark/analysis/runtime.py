from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import DType, RuntimeMemoryEstimate, ArchitectureInfo, ParameterCounts


@dataclass
class KVCacheConfig:
    context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    batch_size: int = 1
    dtype: DType = DType.FP16
    use_mqa: bool = False
    use_gqa: bool = False


def estimate_kv_cache(config: KVCacheConfig) -> Dict[str, float]:
    if config.use_mqa or (config.use_gqa and config.num_kv_heads == 1):
        kv_heads = 1
    else:
        kv_heads = config.num_kv_heads

    elements_per_token = 2 * config.num_layers * kv_heads * config.head_dim
    total_elements = elements_per_token * config.context_length * config.batch_size
    bytes_total = total_elements * config.dtype.bytes_per_element
    gb = bytes_total / (1024 ** 3)

    return {
        "elements_per_token": elements_per_token,
        "total_elements": total_elements,
        "bytes": bytes_total,
        "gb": round(gb, 4),
        "assumptions": {
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "kv_heads_used": kv_heads,
            "dtype": config.dtype.value,
        },
    }


@dataclass
class ActivationConfig:
    batch_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    intermediate_size: Optional[int] = None
    dtype: DType = DType.FP16
    include_attention: bool = True
    include_ffn: bool = True
    checkpointing: bool = False


def estimate_activations(config: ActivationConfig) -> Dict[str, float]:
    activation_bytes = 0

    if config.include_attention:
        qkv_elements = 3 * config.batch_size * config.context_length * config.hidden_size
        attn_out_elements = config.batch_size * config.context_length * config.hidden_size
        activation_bytes += (qkv_elements + attn_out_elements) * config.dtype.bytes_per_element

    if config.include_ffn:
        ffn_size = config.intermediate_size or 4 * config.hidden_size
        gate_up_elements = 2 * config.batch_size * config.context_length * ffn_size
        down_elements = config.batch_size * config.context_length * config.hidden_size
        activation_bytes += (gate_up_elements + down_elements) * config.dtype.bytes_per_element

    activation_bytes *= config.num_layers

    if config.checkpointing:
        activation_bytes = activation_bytes / config.num_layers * 2

    gb = activation_bytes / (1024 ** 3)

    return {
        "bytes": activation_bytes,
        "gb": round(gb, 4),
        "assumptions": {
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "intermediate_size": config.intermediate_size,
            "dtype": config.dtype.value,
            "checkpointing": config.checkpointing,
        },
    }


def estimate_runtime_memory(
    param_count: int,
    weight_dtype: DType,
    architecture: Optional[ArchitectureInfo] = None,
    context_length: int = 2048,
    batch_size: int = 1,
    activation_dtype: DType = DType.FP16,
    kv_cache_dtype: Optional[DType] = None,
    include_activations: bool = True,
    activation_checkpointing: bool = False,
    overhead_factor: float = 0.1,
) -> RuntimeMemoryEstimate:
    weights_bytes = param_count * weight_dtype.bytes_per_element
    weights_gb = weights_bytes / (1024 ** 3)

    kv_dtype = kv_cache_dtype or activation_dtype
    kv_gb = 0.0
    act_gb = 0.0

    if architecture and architecture.num_layers and architecture.hidden_size:
        num_layers = architecture.num_layers
        hidden_size = architecture.hidden_size
        num_kv_heads = architecture.num_kv_heads or architecture.num_attention_heads or 32
        head_dim = hidden_size // (architecture.num_attention_heads or 32)

        kv_config = KVCacheConfig(
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            batch_size=batch_size,
            dtype=kv_dtype,
            use_mqa=architecture.attention_type == "mqa",
            use_gqa=architecture.attention_type == "gqa",
        )
        kv_result = estimate_kv_cache(kv_config)
        kv_gb = kv_result["gb"]

        if include_activations:
            act_config = ActivationConfig(
                batch_size=batch_size,
                context_length=context_length,
                hidden_size=hidden_size,
                num_layers=num_layers,
                intermediate_size=architecture.intermediate_size,
                dtype=activation_dtype,
                checkpointing=activation_checkpointing,
            )
            act_result = estimate_activations(act_config)
            act_gb = act_result["gb"]

    overhead_gb = (weights_gb + kv_gb + act_gb) * overhead_factor
    total_gb = weights_gb + kv_gb + act_gb + overhead_gb

    return RuntimeMemoryEstimate(
        weights_gb=round(weights_gb, 4),
        kv_cache_gb=round(kv_gb, 4),
        activations_gb=round(act_gb, 4),
        overhead_gb=round(overhead_gb, 4),
        total_gb=round(total_gb, 4),
        assumptions={
            "batch_size": batch_size,
            "context_length": context_length,
            "weight_dtype": weight_dtype.value,
            "activation_dtype": activation_dtype.value,
            "kv_cache_dtype": kv_dtype.value,
            "activation_checkpointing": activation_checkpointing,
            "overhead_factor": overhead_factor,
            "note": "Estimates are approximate and architecture-dependent. Actual memory may vary significantly.",
        },
    )


def estimate_memory_for_generation(
    param_count: int,
    weight_dtype: DType,
    architecture: ArchitectureInfo,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    **kwargs,
) -> Dict[str, float]:
    base = estimate_runtime_memory(param_count, weight_dtype, architecture, **kwargs)

    prefill_ctx = kwargs.get("context_length", 2048)
    decode_ctx = prefill_ctx + max_new_tokens

    decode_estimate = estimate_runtime_memory(
        param_count, weight_dtype, architecture,
        context_length=decode_ctx,
        batch_size=batch_size,
        **{k: v for k, v in kwargs.items() if k != "context_length"},
    )

    return {
        "prefill_total_gb": base.total_gb,
        "decode_total_gb": decode_estimate.total_gb,
        "peak_kv_cache_gb": decode_estimate.kv_cache_gb,
        "memory_growth_gb": decode_estimate.total_gb - base.total_gb,
    }


def get_memory_breakdown(estimate: RuntimeMemoryEstimate) -> Dict[str, Any]:
    total = estimate.total_gb
    if total == 0:
        return {}

    return {
        "weights_pct": round(estimate.weights_gb / total * 100, 1),
        "kv_cache_pct": round(estimate.kv_cache_gb / total * 100, 1),
        "activations_pct": round(estimate.activations_gb / total * 100, 1),
        "overhead_pct": round(estimate.overhead_gb / total * 100, 1),
        "components_gb": {
            "weights": estimate.weights_gb,
            "kv_cache": estimate.kv_cache_gb,
            "activations": estimate.activations_gb,
            "overhead": estimate.overhead_gb,
        },
    }