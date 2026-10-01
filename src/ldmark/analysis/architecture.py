from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

from .models import ArchitectureInfo


ARCHITECTURE_PATTERNS = {
    "llama": ["llama", "LlamaForCausalLM", "model.layers", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["mistral", "MistralForCausalLM", "model.layers", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mixtral": ["mixtral", "MixtralForCausalLM", "model.layers", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "block_sparse_moe"],
    "gemma": ["gemma", "GemmaForCausalLM", "model.layers", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["qwen", "QwenForCausalLM", "Qwen2ForCausalLM", "transformer.h", "attn.c_attn", "mlp.c_fc"],
    "phi": ["phi", "PhiForCausalLM", "Phi3ForCausalLM", "model.layers", "self_attn", "qkv_proj", "o_proj"],
    "falcon": ["falcon", "FalconForCausalLM", "transformer.h", "self_attention", "query_key_value", "dense", "mlp.dense_h_to_4h"],
    "mpt": ["mpt", "MPTForCausalLM", "transformer.blocks", "attn.Wqkv", "ffn.up_proj"],
    "bloom": ["bloom", "BloomForCausalLM", "transformer.h", "self_attention", "query_key_value", "dense", "mlp.dense_h_to_4h"],
    "gpt_neox": ["gpt_neox", "GPTNeoXForCausalLM", "gpt_neox.layers", "attention.query_key_value", "mlp.dense_h_to_4h"],
    "gpt_j": ["gpt_j", "GPTJForCausalLM", "transformer.h", "attn.k_proj", "attn.v_proj", "attn.q_proj", "mlp.fc_in", "mlp.fc_out"],
    "opt": ["opt", "OPTForCausalLM", "model.decoder.layers", "self_attn.q_proj", "fc1", "fc2"],
    "bert": ["bert", "BertForMaskedLM", "encoder.layer", "attention.self.query", "intermediate.dense"],
    "roberta": ["roberta", "RobertaForMaskedLM", "encoder.layer", "attention.self.query", "intermediate.dense"],
    "t5": ["t5", "T5ForConditionalGeneration", "encoder.block", "layer.0.SelfAttention", "layer.1.DenseReluDense"],
}


ATTENTION_TYPE_KEYWORDS = {
    "mha": ["multi_head_attention", "multiheadattention", "mha"],
    "mqa": ["multi_query_attention", "multiqueryattention", "mqa"],
    "gqa": ["grouped_query_attention", "groupedqueryattention", "gqa"],
    "sliding_window": ["sliding_window", "window_attention"],
    "flash": ["flash_attention", "flash_attn"],
}


def detect_architecture_from_config(config: Dict[str, Any]) -> Optional[str]:
    arch_list = config.get("architectures", [])
    if arch_list:
        for arch in arch_list:
            for canonical, patterns in ARCHITECTURE_PATTERNS.items():
                if any(p.lower() in arch.lower() for p in patterns):
                    return canonical

    model_type = config.get("model_type", "").lower()
    for canonical, patterns in ARCHITECTURE_PATTERNS.items():
        if any(p.lower() in model_type for p in patterns):
            return canonical

    return None


def detect_architecture_from_tensor_names(tensor_names: List[str]) -> Optional[str]:
    name_str = " ".join(tensor_names).lower()
    for canonical, patterns in ARCHITECTURE_PATTERNS.items():
        if any(p.lower() in name_str for p in patterns):
            return canonical
    return None


def extract_attention_type(config: Dict[str, Any], tensor_names: List[str]) -> Optional[str]:
    name_str = " ".join(tensor_names).lower()
    config_str = str(config).lower()

    for attn_type, keywords in ATTENTION_TYPE_KEYWORDS.items():
        if any(k in name_str or k in config_str for k in keywords):
            return attn_type

    num_heads = config.get("num_attention_heads") or config.get("n_head")
    num_kv_heads = config.get("num_key_value_heads") or config.get("n_kv_head")
    if num_heads and num_kv_heads:
        if num_kv_heads == 1:
            return "mqa"
        elif num_kv_heads < num_heads:
            return "gqa"
        else:
            return "mha"
    return None


def build_architecture_info(
    config: Dict[str, Any],
    tensor_names: List[str],
    parameter_counts: Optional[Dict[str, int]] = None,
) -> ArchitectureInfo:
    arch = detect_architecture_from_config(config)
    if not arch:
        arch = detect_architecture_from_tensor_names(tensor_names)

    attention_type = extract_attention_type(config, tensor_names)

    num_layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    hidden_size = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    intermediate_size = config.get("intermediate_size") or config.get("n_inner") or config.get("ffn_dim")
    num_attention_heads = config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads")
    num_kv_heads = config.get("num_key_value_heads") or config.get("n_kv_head")
    vocab_size = config.get("vocab_size") or config.get("n_vocab")
    max_context_length = config.get("max_position_embeddings") or config.get("n_positions") or config.get("max_seq_len") or config.get("context_length")

    additional = {}
    if parameter_counts:
        additional["parameter_breakdown"] = parameter_counts

    for key, value in config.items():
        if key not in [
            "architectures", "model_type", "num_hidden_layers", "n_layer", "num_layers",
            "hidden_size", "n_embd", "d_model", "intermediate_size", "n_inner", "ffn_dim",
            "num_attention_heads", "n_head", "num_heads", "num_key_value_heads", "n_kv_head",
            "vocab_size", "n_vocab", "max_position_embeddings", "n_positions", "max_seq_len", "context_length",
            "tie_word_embeddings",
        ]:
            additional[key] = value

    return ArchitectureInfo(
        model_architecture=arch,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
        attention_type=attention_type,
        additional_metadata=additional,
    )


def infer_architecture_from_state_dict(state_dict: Dict[str, Any]) -> ArchitectureInfo:
    tensor_names = list(state_dict.keys())
    return build_architecture_info({}, tensor_names)