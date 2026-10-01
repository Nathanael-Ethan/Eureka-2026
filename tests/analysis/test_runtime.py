import pytest
from src.ldmark.analysis.runtime import (
    KVCacheConfig,
    ActivationConfig,
    estimate_kv_cache,
    estimate_activations,
    estimate_runtime_memory,
    estimate_memory_for_generation,
    get_memory_breakdown,
)
from src.ldmark.analysis.models import DType, ArchitectureInfo


class TestEstimateKVCache:
    def test_basic(self):
        config = KVCacheConfig(
            context_length=2048,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            batch_size=1,
            dtype=DType.FP16,
        )
        result = estimate_kv_cache(config)
        assert result["gb"] > 0
        assert result["elements_per_token"] == 2 * 32 * 32 * 128

    def test_mqa(self):
        config = KVCacheConfig(
            context_length=2048,
            num_layers=32,
            num_kv_heads=1,
            head_dim=128,
            batch_size=1,
            dtype=DType.FP16,
            use_mqa=True,
        )
        result = estimate_kv_cache(config)
        assert result["assumptions"]["kv_heads_used"] == 1

    def test_gqa(self):
        config = KVCacheConfig(
            context_length=2048,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            batch_size=1,
            dtype=DType.FP16,
            use_gqa=True,
        )
        result = estimate_kv_cache(config)
        assert result["assumptions"]["kv_heads_used"] == 8

    def test_batch_size(self):
        config1 = KVCacheConfig(context_length=2048, num_layers=32, num_kv_heads=32, head_dim=128, batch_size=1)
        config2 = KVCacheConfig(context_length=2048, num_layers=32, num_kv_heads=32, head_dim=128, batch_size=4)
        result1 = estimate_kv_cache(config1)
        result2 = estimate_kv_cache(config2)
        assert result2["gb"] == pytest.approx(result1["gb"] * 4, rel=0.01)


class TestEstimateActivations:
    def test_basic(self):
        config = ActivationConfig(
            batch_size=1,
            context_length=2048,
            hidden_size=4096,
            num_layers=32,
            intermediate_size=11008,
            dtype=DType.FP16,
        )
        result = estimate_activations(config)
        assert result["gb"] > 0

    def test_checkpointing(self):
        config1 = ActivationConfig(
            batch_size=1, context_length=2048, hidden_size=4096, num_layers=32, intermediate_size=11008, checkpointing=False
        )
        config2 = ActivationConfig(
            batch_size=1, context_length=2048, hidden_size=4096, num_layers=32, intermediate_size=11008, checkpointing=True
        )
        result1 = estimate_activations(config1)
        result2 = estimate_activations(config2)
        assert result2["gb"] < result1["gb"]


class TestEstimateRuntimeMemory:
    def test_full_architecture(self):
        arch = ArchitectureInfo(
            num_layers=32,
            hidden_size=4096,
            intermediate_size=11008,
            num_attention_heads=32,
            num_kv_heads=8,
        )
        est = estimate_runtime_memory(
            param_count=7_000_000_000,
            weight_dtype=DType.FP16,
            architecture=arch,
            context_length=2048,
            batch_size=1,
        )
        assert est.weights_gb > 0
        assert est.kv_cache_gb > 0
        assert est.activations_gb > 0
        assert est.total_gb == est.weights_gb + est.kv_cache_gb + est.activations_gb + est.overhead_gb

    def test_no_architecture(self):
        est = estimate_runtime_memory(
            param_count=7_000_000_000,
            weight_dtype=DType.FP16,
            architecture=None,
        )
        assert est.weights_gb > 0
        assert est.kv_cache_gb == 0
        assert est.activations_gb == 0


class TestGetMemoryBreakdown:
    def test_breakdown(self):
        est = estimate_runtime_memory(
            param_count=7_000_000_000,
            weight_dtype=DType.FP16,
            architecture=ArchitectureInfo(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8),
            context_length=2048,
        )
        breakdown = get_memory_breakdown(est)
        assert "weights_pct" in breakdown
        assert "kv_cache_pct" in breakdown
        assert "activations_pct" in breakdown
        assert "overhead_pct" in breakdown
        total_pct = breakdown["weights_pct"] + breakdown["kv_cache_pct"] + breakdown["activations_pct"] + breakdown["overhead_pct"]
        assert total_pct == pytest.approx(100.0, abs=0.5)


class TestEstimateMemoryForGeneration:
    def test_generation(self):
        arch = ArchitectureInfo(num_layers=32, hidden_size=4096, num_attention_heads=32, num_kv_heads=8)
        result = estimate_memory_for_generation(
            param_count=7_000_000_000,
            weight_dtype=DType.FP16,
            architecture=arch,
            max_new_tokens=100,
            context_length=2048,
        )
        assert "prefill_total_gb" in result
        assert "decode_total_gb" in result
        assert "peak_kv_cache_gb" in result
        assert result["decode_total_gb"] > result["prefill_total_gb"]