import pytest
from src.ldmark.analysis.architecture import (
    detect_architecture_from_config,
    detect_architecture_from_tensor_names,
    build_architecture_info,
    infer_architecture_from_state_dict,
    extract_attention_type,
)
from src.ldmark.analysis.models import ArchitectureInfo


class TestDetectArchitectureFromConfig:
    def test_llama_architecture(self):
        config = {"architectures": ["LlamaForCausalLM"]}
        arch = detect_architecture_from_config(config)
        assert arch == "llama"

    def test_mistral_model_type(self):
        config = {"model_type": "mistral"}
        arch = detect_architecture_from_config(config)
        assert arch == "mistral"

    def test_qwen2(self):
        config = {"architectures": ["Qwen2ForCausalLM"]}
        arch = detect_architecture_from_config(config)
        assert arch == "qwen"

    def test_unknown(self):
        config = {"model_type": "unknown_model"}
        arch = detect_architecture_from_config(config)
        assert arch is None


class TestDetectArchitectureFromTensorNames:
    def test_from_tensor_names(self):
        names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"]
        arch = detect_architecture_from_tensor_names(names)
        assert arch == "llama"

    def test_mistral_names(self):
        names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"]
        arch = detect_architecture_from_tensor_names(names)
        assert arch in ["llama", "mistral"]

    def test_unknown_names(self):
        names = ["unknown.layer.weight"]
        arch = detect_architecture_from_tensor_names(names)
        assert arch is None


class TestExtractAttentionType:
    def test_mha_from_config(self):
        config = {"num_attention_heads": 32, "num_key_value_heads": 32}
        names = []
        attn = extract_attention_type(config, names)
        assert attn == "mha"

    def test_mqa_from_config(self):
        config = {"num_attention_heads": 32, "num_key_value_heads": 1}
        names = []
        attn = extract_attention_type(config, names)
        assert attn == "mqa"

    def test_gqa_from_config(self):
        config = {"num_attention_heads": 32, "num_key_value_heads": 8}
        names = []
        attn = extract_attention_type(config, names)
        assert attn == "gqa"

    def test_flash_attention_from_names(self):
        config = {}
        names = ["model.layers.0.self_attn.flash_attn.q_proj.weight"]
        attn = extract_attention_type(config, names)
        assert attn == "flash"


class TestBuildArchitectureInfo:
    def test_full_config(self):
        config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        }
        names = ["model.layers.0.self_attn.q_proj.weight"]
        arch = build_architecture_info(config, names)
        assert arch.model_architecture == "llama"
        assert arch.num_layers == 32
        assert arch.hidden_size == 4096
        assert arch.intermediate_size == 11008
        assert arch.num_attention_heads == 32
        assert arch.num_kv_heads == 8
        assert arch.vocab_size == 32000
        assert arch.max_context_length == 4096
        assert arch.attention_type == "gqa"

    def test_minimal_config(self):
        config = {"hidden_size": 256, "num_hidden_layers": 4}
        names = []
        arch = build_architecture_info(config, names)
        assert arch.hidden_size == 256
        assert arch.num_layers == 4
        assert arch.model_architecture is None


class TestInferArchitectureFromStateDict:
    def test_infer(self):
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": None,
            "model.layers.0.mlp.gate_proj.weight": None,
        }
        arch = infer_architecture_from_state_dict(state_dict)
        assert arch.model_architecture == "llama"