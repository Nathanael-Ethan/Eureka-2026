#!/usr/bin/env python3
"""
Create a small real model fixture for LDMARK end-to-end testing.

This creates a tiny transformer model (similar to Llama architecture but very small)
that is structurally realistic and can be used for real compilation experiments.
"""

import numpy as np
import json
from pathlib import Path


def create_tiny_llama_model(
    num_layers: int = 2,
    hidden_size: int = 128,
    intermediate_size: int = 512,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    vocab_size: int = 1024,
    max_position_embeddings: int = 512,
    dtype: np.dtype = np.float16,
    seed: int = 42,
) -> dict:
    """
    Create a tiny Llama-like model with realistic tensor structure.
    
    Returns a dict with 'tensors' and 'config' keys.
    """
    np.random.seed(seed)
    
    tensors = {}
    head_dim = hidden_size // num_heads
    
    # Embedding layer
    tensors["model.embed_tokens.weight"] = np.random.randn(vocab_size, hidden_size).astype(dtype) * 0.02
    
    # Transformer layers
    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        
        # Attention projections
        # QKV combined projection
        qkv_out_dim = (num_heads + 2 * num_kv_heads) * head_dim
        tensors[f"{prefix}.self_attn.qkv_proj.weight"] = np.random.randn(qkv_out_dim, hidden_size).astype(dtype) * 0.02
        
        # Output projection
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(dtype) * 0.02
        
        # MLP projections
        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(intermediate_size, hidden_size).astype(dtype) * 0.02
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(intermediate_size, hidden_size).astype(dtype) * 0.02
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(hidden_size, intermediate_size).astype(dtype) * 0.02
        
        # Layer norms
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hidden_size, dtype=dtype)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hidden_size, dtype=dtype)
    
    # Final norm
    tensors["model.norm.weight"] = np.ones(hidden_size, dtype=dtype)
    
    # LM head (tied to embeddings in some models, separate here)
    tensors["lm_head.weight"] = np.random.randn(vocab_size, hidden_size).astype(dtype) * 0.02
    
    # Config matching the model structure
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    
    return {"tensors": tensors, "config": config}


def save_model_npz(model_dict: dict, output_path: Path) -> None:
    """Save model tensors as .npz file with config as JSON sidecar."""
    tensors = model_dict["tensors"]
    config = model_dict["config"]
    
    # Save tensors
    np.savez_compressed(output_path, **tensors)
    
    # Save config
    config_path = output_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved model to {output_path}")
    print(f"Saved config to {config_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.2f} KB")


def save_model_safetensors(model_dict: dict, output_path: Path) -> None:
    """Save model tensors as .safetensors file with config as JSON sidecar."""
    try:
        from safetensors.numpy import save_file
    except ImportError:
        print("safetensors not available, skipping .safetensors save")
        return
    
    tensors = model_dict["tensors"]
    config = model_dict["config"]
    
    # Save tensors
    save_file(tensors, output_path)
    
    # Save config
    config_path = output_path.with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved model to {output_path}")
    print(f"Saved config to {config_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.2f} KB")


def main():
    output_dir = Path(__file__).parent
    
    # Create the tiny model
    model = create_tiny_llama_model(
        num_layers=2,
        hidden_size=128,
        intermediate_size=512,
        num_heads=4,
        num_kv_heads=4,
        vocab_size=1024,
        max_position_embeddings=512,
        dtype=np.float16,
        seed=42,
    )
    
    # Calculate parameter count
    total_params = sum(t.size for t in model["tensors"].values())
    print(f"Total parameters: {total_params:,}")
    print(f"Tensor count: {len(model['tensors'])}")
    
    for name, tensor in model["tensors"].items():
        print(f"  {name}: {tensor.shape} ({tensor.dtype}) = {tensor.size:,} params")
    
    # Save as NPZ (always available)
    npz_path = output_dir / "tiny_llama_2L_128H.npz"
    save_model_npz(model, npz_path)
    
    # Save as safetensors if available
    st_path = output_dir / "tiny_llama_2L_128H.safetensors"
    save_model_safetensors(model, st_path)


if __name__ == "__main__":
    main()