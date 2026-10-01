from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from collections import Counter

from .models import DType, ParameterCounts, TensorInfo, TensorTrainableStatus


def count_parameters(tensors: List[TensorInfo]) -> ParameterCounts:
    counts = ParameterCounts()
    for tensor in tensors:
        counts.add_tensor(tensor.name, tensor.num_elements, tensor.dtype, tensor.trainable)
    return counts


def count_parameters_from_dict(tensor_dict: Dict[str, Any]) -> ParameterCounts:
    from .tensor import analyze_torch_state_dict
    tensors = analyze_torch_state_dict(tensor_dict)
    return count_parameters(tensors)


def count_parameters_from_numpy(tensor_dict: Dict[str, Any]) -> ParameterCounts:
    from .tensor import analyze_numpy_dict
    tensors = analyze_numpy_dict(tensor_dict)
    return count_parameters(tensors)


def get_parameter_summary(counts: ParameterCounts) -> Dict[str, Any]:
    return {
        "total_parameters": counts.total,
        "trainable_parameters": counts.trainable,
        "frozen_parameters": counts.non_trainable,
        "unknown_trainable_parameters": counts.unknown_trainable,
        "by_dtype": {k.value: v for k, v in counts.by_dtype.items()},
        "num_tensors": len(counts.by_tensor),
        "largest_tensors": sorted(counts.by_tensor.items(), key=lambda x: x[1], reverse=True)[:10],
    }


def estimate_parameters_from_config(config: Dict[str, Any]) -> Optional[int]:
    arch = config.get("architectures", [None])[0] if config.get("architectures") else None
    arch = arch or config.get("model_type", "").lower()

    hidden_size = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    num_layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    intermediate_size = config.get("intermediate_size") or config.get("n_inner") or config.get("ffn_dim")
    num_heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads") or 32
    vocab_size = config.get("vocab_size") or config.get("n_vocab")
    num_kv_heads = config.get("num_key_value_heads") or config.get("n_kv_head") or num_heads

    if not all([hidden_size, num_layers]):
        return None

    head_dim = hidden_size // num_heads
    qkv_params = 3 * hidden_size * head_dim * num_heads
    if num_kv_heads and num_kv_heads != num_heads:
        qkv_params = hidden_size * head_dim * (num_heads + 2 * num_kv_heads)

    attn_out_params = hidden_size * hidden_size
    ffn_params = 2 * hidden_size * (intermediate_size or 4 * hidden_size)
    layer_params = qkv_params + attn_out_params + ffn_params

    embed_params = vocab_size * hidden_size if vocab_size else 0
    norm_params = 2 * hidden_size * num_layers
    lm_head_params = vocab_size * hidden_size if vocab_size and config.get("tie_word_embeddings", True) == False else 0

    total = embed_params + num_layers * layer_params + norm_params + lm_head_params
    return total


def compare_parameter_counts(actual: ParameterCounts, estimated: int) -> Dict[str, Any]:
    diff = actual.total - estimated
    pct_diff = (diff / estimated * 100) if estimated > 0 else 0
    return {
        "actual_total": actual.total,
        "estimated_total": estimated,
        "difference": diff,
        "percent_difference": round(pct_diff, 2),
        "match": abs(pct_diff) < 5.0,
    }