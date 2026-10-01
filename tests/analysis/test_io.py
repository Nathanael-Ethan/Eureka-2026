import pytest
import numpy as np
from src.ldmark.analysis.io import (
    create_synthetic_model,
    create_synthetic_state_dict,
    _factorize_size,
)


class TestCreateSyntheticModel:
    def test_basic_model(self):
        model = create_synthetic_model(
            num_layers=2,
            hidden_size=128,
            intermediate_size=512,
            num_heads=4,
            vocab_size=1000,
        )
        assert "tensors" in model
        assert "config" in model
        assert len(model["tensors"]) > 0
        assert model["config"]["num_hidden_layers"] == 2
        assert model["config"]["hidden_size"] == 128
        assert model["config"]["num_attention_heads"] == 4

    def test_tensor_shapes(self):
        model = create_synthetic_model(num_layers=1, hidden_size=64, num_heads=2, vocab_size=100)
        tensors = model["tensors"]
        assert "model.embed_tokens.weight" in tensors
        assert tensors["model.embed_tokens.weight"].shape == (100, 64)
        assert "model.layers.0.self_attn.qkv_proj.weight" in tensors
        assert "lm_head.weight" in tensors

    def test_dtype(self):
        model = create_synthetic_model(dtype="float32")
        for tensor in model["tensors"].values():
            assert tensor.dtype == np.float32

        model = create_synthetic_model(dtype="float16")
        for tensor in model["tensors"].values():
            assert tensor.dtype == np.float16


class TestCreateSyntheticStateDict:
    def test_parameter_count(self):
        state_dict = create_synthetic_state_dict(param_count=1_000_000, num_tensors=10)
        total = sum(t.size for t in state_dict.values())
        assert total == 1_000_000

    def test_num_tensors(self):
        state_dict = create_synthetic_state_dict(param_count=1_000_000, num_tensors=5)
        assert len(state_dict) == 5

    def test_dtype(self):
        state_dict = create_synthetic_state_dict(dtype="float32")
        for tensor in state_dict.values():
            assert tensor.dtype == np.float32


class TestFactorizeSize:
    def test_perfect_square(self):
        assert _factorize_size(100) == (10, 10)

    def test_prime(self):
        assert _factorize_size(17) == (17, 1)

    def test_even(self):
        assert _factorize_size(1000)[0] * _factorize_size(1000)[1] == 1000

    def test_one(self):
        assert _factorize_size(1) == (1,)