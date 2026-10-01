from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    with open(path, "r") as f:
        return json.load(f)


def load_safetensors_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            tensors = {}
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensors[key] = tensor
            return {"metadata": metadata, "tensors": tensors}
    except ImportError:
        raise ImportError("safetensors package required. Install with: pip install safetensors")


def load_pytorch_state_dict(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def load_numpy_dict(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    import numpy as np
    return dict(np.load(path))


def save_analysis_json(analysis: Any, file_path: Union[str, Path], indent: int = 2):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if hasattr(analysis, "to_json"):
            f.write(analysis.to_json(indent))
        else:
            json.dump(analysis, f, indent=indent)


def load_analysis_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(file_path)
    with open(path, "r") as f:
        return json.load(f)


def create_synthetic_model(
    num_layers: int = 4,
    hidden_size: int = 256,
    intermediate_size: int = 1024,
    num_heads: int = 8,
    vocab_size: int = 32000,
    dtype: str = "float16",
) -> Dict[str, Any]:
    import numpy as np
    from .tensor import DTYPE_TO_NUMPY, infer_dtype_from_name

    np_dtype = DTYPE_TO_NUMPY.get(infer_dtype_from_name("", dtype), np.float16)

    tensors = {}
    head_dim = hidden_size // num_heads

    tensors["model.embed_tokens.weight"] = np.random.randn(vocab_size, hidden_size).astype(np_dtype)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        qkv_size = 3 * hidden_size * head_dim * num_heads
        tensors[f"{prefix}.self_attn.qkv_proj.weight"] = np.random.randn(3 * num_heads * head_dim, hidden_size).astype(np_dtype)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(np_dtype)

        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(intermediate_size, hidden_size).astype(np_dtype)
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(intermediate_size, hidden_size).astype(np_dtype)
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(hidden_size, intermediate_size).astype(np_dtype)

        tensors[f"{prefix}.input_layernorm.weight"] = np.random.randn(hidden_size).astype(np_dtype)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.random.randn(hidden_size).astype(np_dtype)

    tensors["model.norm.weight"] = np.random.randn(hidden_size).astype(np_dtype)
    tensors["lm_head.weight"] = np.random.randn(vocab_size, hidden_size).astype(np_dtype)

    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": 2048,
    }

    return {"tensors": tensors, "config": config}


def create_synthetic_state_dict(
    param_count: int = 1000000,
    num_tensors: int = 10,
    dtype: str = "float16",
) -> Dict[str, Any]:
    import numpy as np
    from .tensor import DTYPE_TO_NUMPY, infer_dtype_from_name

    np_dtype = DTYPE_TO_NUMPY.get(infer_dtype_from_name("", dtype), np.float16)

    elements_per_tensor = param_count // num_tensors
    tensors = {}

    for i in range(num_tensors):
        remaining = param_count - sum(t.size for t in tensors.values())
        if i == num_tensors - 1:
            size = remaining
        else:
            size = min(elements_per_tensor, remaining // (num_tensors - i))

        shape = _factorize_size(size)
        tensors[f"layer.{i}.weight"] = np.random.randn(*shape).astype(np_dtype)

    return tensors


def _factorize_size(size: int) -> tuple:
    if size <= 1:
        return (1,)
    for i in range(int(size**0.5), 0, -1):
        if size % i == 0:
            return (size // i, i)
    return (size, 1)