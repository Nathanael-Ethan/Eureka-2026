"""M3: laptop-budget runtime — mmap/lazy load, peak-RSS, budget enforcement."""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

import numpy as np
import pytest

from src.ldmark.artifact.writer import LDMARKArtifactWriter
from src.ldmark.artifact.format import (
    ArtifactFormatVersion, ModelInfo, TensorEncoding, QuantizationParams,
)
from src.ldmark.compression.quantize import quantize_int8
from src.ldmark.runtime import open_runtime
from src.ldmark.runtime.memory import get_process_rss_bytes
from src.ldmark.runtime.exceptions import MemoryAccountingError
from src.ldmark.hardware import (
    laptop_cpu_profile_4gb, laptop_cpu_profile_8gb,
    apple_silicon_cpu_profile, laptop_budget_4gb, laptop_budget_8gb,
    check_runtime_against_budget,
)
from src.ldmark.strategy.feasibility import check_plan_against_laptop_budget


def _make_artifact(path: str, n: int = 4, shape=(128, 256)) -> None:
    w = LDMARKArtifactWriter(path, ArtifactFormatVersion.V1, overwrite=True)
    w.set_model_info(ModelInfo(
        model_id="tiny-llama-demo", architecture="llama",
        parameter_count=int(np.prod(shape)) * n, tensor_count=n,
        num_layers=2, hidden_size=128, original_dtype="float16"))
    for i in range(n):
        t = np.random.randn(*shape).astype(np.float32) * 0.1
        q = quantize_int8(t, group_size=128)
        w.add_tensor(
            name=f"layer.{i}.weight", data=q.data, scales=q.scales,
            original_shape=t.shape, original_dtype=t.dtype,
            encoding=TensorEncoding.GROUPWISE_QUANTIZED,
            quantization=QuantizationParams(target_bits=8, group_size=128, scale_dtype="float16"))
    w.write()


def test_lazy_mmap_no_payload_on_open():
    with tempfile.TemporaryDirectory() as td:
        ad = os.path.join(td, "a")
        _make_artifact(ad)
        with open_runtime(ad, mmap_tensors=True) as rt:
            assert len(rt._loaded_tensors) == 0
            mem = rt.get_memory_usage()
            assert mem["compressed_weight_bytes"] == 0
            assert mem["decompressed_weight_bytes"] == 0
            assert len(rt.list_tensors()) == 4


def test_storage_vs_runtime_separated_and_rss():
    with tempfile.TemporaryDirectory() as td:
        ad = os.path.join(td, "a")
        _make_artifact(ad)
        with open_runtime(ad, mmap_tensors=True) as rt:
            t0 = time.perf_counter()
            for name in rt.list_tensors():
                rt.get_tensor(name, decompress=True)
            load_ms = (time.perf_counter() - t0) * 1000
            mem = rt.get_memory_usage()
            # storage (compressed INT8) must be much smaller than runtime (fp16)
            assert mem["compressed_weight_bytes"] > 0
            assert mem["decompressed_weight_bytes"] > mem["compressed_weight_bytes"]
            rss = rt.get_peak_rss_bytes()
            assert rss > 0
            assert get_process_rss_bytes() > 0
            # full-decompressed estimate matches fp16 weight bytes
            est = rt.estimated_full_decompressed_bytes()
            assert est == 4 * 128 * 256 * 2
            assert load_ms > 0


def test_budget_accept_and_reject():
    with tempfile.TemporaryDirectory() as td:
        ad = os.path.join(td, "a")
        _make_artifact(ad)
        with open_runtime(ad) as rt:
            verdict = rt.check_budget(4 * 1024 ** 3, label="tiny_llama demo")
            assert verdict["fits"] is True
            with pytest.raises(MemoryAccountingError, match="REJECTED"):
                rt.check_budget(1024, label="tiny_llama demo")


def test_laptop_profiles_and_budgets():
    p4 = laptop_cpu_profile_4gb()
    p8 = laptop_cpu_profile_8gb()
    m1 = apple_silicon_cpu_profile()
    assert p4.system.total_ram_bytes == 4 * 1024 ** 3
    assert p8.system.total_ram_bytes == 8 * 1024 ** 3
    assert m1.system.total_ram_bytes == 8 * 1024 ** 3
    assert p4.gpu is None  # CPU-only
    b4 = laptop_budget_4gb()
    b8 = laptop_budget_8gb()
    assert b8.max_runtime_memory_bytes == 4 * 1024 ** 3
    assert b4.max_runtime_memory_bytes == 2 * 1024 ** 3
    # tiny model fits; huge model rejected with clear message
    ok = check_runtime_against_budget(262_144, b8, label="tiny_llama demo")
    assert ok.fits_runtime_budget is True
    bad = check_runtime_against_budget(16 * 1024 ** 3, b8, label="big plan")
    assert bad.fits_runtime_budget is False
    assert any("REJECTED" in w for w in bad.warnings)
    with pytest.raises(ValueError, match="REJECTED"):
        check_plan_against_laptop_budget(
            param_count=8_000_000_000, max_runtime_memory_bytes=b8.max_runtime_memory_bytes,
            label="8B fp16 plan")
    verdict = check_plan_against_laptop_budget(
        param_count=787_000, max_runtime_memory_bytes=b8.max_runtime_memory_bytes,
        label="tiny_llama demo")
    assert verdict["fits"] is True
