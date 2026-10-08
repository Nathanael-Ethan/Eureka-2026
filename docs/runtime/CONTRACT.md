# LDMARK Runtime Format Contract

**Version:** 1.0  
**Status:** Experimental  
**Last Updated:** 2024

---

## Overview

This document defines the contract between LDMARK artifacts and the LDMARK Runtime.
It specifies supported artifact versions, compression formats, tensor loading behavior,
memory behavior, reconstruction behavior, and unsupported features.

The contract is designed to allow future replacement of the NumPy implementation
with optimized kernels (CUDA, ROCm, Metal, etc.) without breaking artifact compatibility.

---

## 1. Supported Artifact Versions

| Version | Identifier | Status | Notes |
|---------|------------|--------|-------|
| 1.0 | `ldmark-artifact-1.0` | ✅ Supported | Current stable format |

**Future versions** must maintain backward compatibility for tensor index entries.
Runtime SHOULD reject unsupported versions with `VersionMismatchError`.

---

## 2. Supported Compression Formats

### 2.1 INT8 (Group-wise Quantization)

- **Encoding:** `groupwise_quantized`
- **Target bits:** 8
- **Scale dtype:** float16 (default), float32
- **Group size:** Configurable (default 128)
- **Symmetric:** true (default), false supported
- **Storage:** int8 per weight + float16 scale per group

**Dequantization:**
```
dequantized[g*group_size:(g+1)*group_size] = int8_data[g*group_size:(g+1)*group_size] * float16_scale[g]
```

### 2.2 INT4 (Group-wise Quantization, Packed)

- **Encoding:** `groupwise_quantized`
- **Target bits:** 4
- **Scale dtype:** float16 (default), float32
- **Group size:** Configurable (default 128)
- **Symmetric:** true (default), false supported
- **Storage:** 2 INT4 values packed per uint8 + float16 scale per group

**Dequantization:**
```
unpacked = unpack_int4(packed_data)  # Sign-extend 4-bit to int8
dequantized = unpacked.astype(float32) * float16_scale[g]
```

### 2.3 Binary (1-bit, Packed)

- **Encoding:** `binary_packed` or `groupwise_quantized` (target_bits=1)
- **Target bits:** 1
- **Scale dtype:** float16 (default), float32
- **Group size:** Configurable (default 128)
- **Storage:** 1 bit per weight (packed 8 weights per byte) + float16 scale per group

**Encoding:** Sign encoding (0 → -1, 1 → +1)

**Dequantization:**
```
bit = (packed_data[i//8] >> (i%8)) & 1
value = scale * (1.0 if bit else -1.0)
```

### 2.4 Ternary (2-bit, Packed)

- **Encoding:** `ternary_packed` or `groupwise_quantized` (target_bits=2)
- **Target bits:** 2 (log2(3) ≈ 1.585 effective)
- **Scale dtype:** float16 (default), float32
- **Group size:** Configurable (default 128)
- **Storage:** 2 bits per weight (packed 4 weights per byte) + float16 scale per group

**Encoding:** 00 → -1, 01 → 0, 10 → +1 (11 unused)

**Dequantization:**
```
trit = (packed_data[byte_idx] >> bit_offset) & 0x3
if trit == 0: value = -scale
elif trit == 1: value = 0.0
else: value = +scale
```

---

## 3. Unsupported Compression Formats

The following formats are defined in the artifact specification but **NOT supported** by this runtime:

| Format | Artifact Encoding | Reason |
|--------|-------------------|--------|
| FP16 (raw) | `raw` | Not compressed - not the focus of this runtime |
| FP8 | `groupwise_quantized` (target_bits=8, different scale) | Not implemented |
| PrismML Q1_0_g128 | `groupwise_quantized` (target_bits=1, special) | Research format |

**Error Behavior:** Attempting to load an unsupported format raises `UnsupportedRuntimeFormat` with the format name and reason.

**DO NOT** silently fall back to FP32. The runtime must fail clearly.

---

## 4. Tensor Loading Behavior

### 4.1 Lazy Loading (MANDATORY)

- `runtime.open()` / `LDMARKRuntime(artifact_dir)` — **MUST NOT** load tensor payloads
- `runtime.get_tensor(name)` — Loads ONLY the requested tensor
- `runtime.get_tensor(name, decompress=True)` — Loads AND decompresses
- `runtime.get_tensor_data(name)` — Loads and decompresses in one call
- `mmap_tensors=True` — Memory-maps tensor payloads (used for large models);
  `open()` still MUST NOT fault payloads in.

### 4.1b Laptop Budget Enforcement (M3)

- `runtime.estimated_full_decompressed_bytes()` — Estimated RUNTIME (decompressed
  fp16/fp32 weights) if all tensors were expanded. This is NOT storage size.
- `runtime.check_budget(max_runtime_memory_bytes, label)` — Returns a `fits`
  verdict dict when within budget; raises `MemoryAccountingError` with a
  `REJECTED: ...` message (estimated runtime, budget, overage) when over.
- `hardware.laptop_budget_4gb()` / `laptop_budget_8gb()` — Predefined laptop
  budgets (2GB / 4GB runtime caps, CPU-only, M1 baseline = 8GB).
- `hardware.laptop_cpu_profile_4gb/8gb()` / `apple_silicon_cpu_profile()` —
  CPU-only profiles; `gpu is None` by design (no GPU execution assumed).
- `strategy.feasibility.check_plan_against_laptop_budget(...)` — Rejects
  over-budget plans with `ValueError("REJECTED: ...")`; storage (compressed)
  vs runtime (decompressed) kept separate.
- `runtime.get_peak_rss_bytes()` / `memory.get_process_rss_bytes()` — MEASURED
  OS peak RSS readback, distinct from estimated counters.

### 4.2 Tensor Lifecycle

1. **Not loaded** — No memory allocated
2. **Loaded (compressed)** — Compressed bytes in memory (tracked by `compressed_weight_bytes`)
3. **Decompressed** — Full FP16/FP32 tensor in memory (tracked by `decompressed_weight_bytes`)
4. **Released** — Memory freed, counters decremented

### 4.3 Tensor Data Structure

Loaded tensor provides:
- `compressed_data: np.ndarray` — Raw bytes from artifact (uint8)
- `scales: Optional[np.ndarray]` — Scale factors (float16) or None
- `entry: TensorIndexEntry` — Full metadata from manifest
- `decompress(target_dtype)` — Explicit dequantization
- `get_data(target_dtype)` — Get data, auto-decompress if needed
- `release_decompressed()` — Free decompressed memory

---

## 5. Memory Behavior

### 5.1 Memory Categories (DISTINGUISHED)

| Category | Variable | Description |
|----------|----------|-------------|
| Compressed | `compressed_weight_bytes` | Bytes of compressed tensor data in memory |
| Decompressed | `decompressed_weight_bytes` | Bytes of reconstructed FP16/FP32 tensors |
| Temporary | `temporary_bytes` | Memory for active computation (matmul inputs/outputs) |
| Peak Temporary | `peak_temporary_bytes` | Maximum temporary memory observed |

### 5.2 Memory Accounting Rules

1. **Compressed bytes** counted when tensor loaded via `get_tensor()`
2. **Decompressed bytes** counted when `decompress()` called
3. **Temporary bytes** counted during `matmul()` operations
4. All counters decremented on `release_tensor()` or `release_all()`
4. **Estimated peak** = max observed `(compressed + decompressed + temporary)`

### 5.3 Model-Scale Thinking

The runtime MUST make the distinction between **storage compression** and **runtime decompression** obvious:

```python
# Example: 27B parameter model at INT4
compressed = 4 GB      # On disk / in artifact
decompressed = 54 GB   # If fully expanded to FP16
```

**Critical:** This runtime does NOT implement direct low-bit computation.
Tensors are fully expanded to FP16/FP32 for computation.
Future work: direct computation on compressed representations.

---

## 6. Reconstruction Behavior

### 6.1 Dequantization Validation

Each dequantization validates:
- ✅ Shape matches `original_shape` in manifest
- ✅ Dtype matches `original_dtype` in manifest
- ✅ Group size matches quantization params
- ✅ Scale count matches expected groups
- ✅ Packed data size matches expected bytes

**Failure:** Raises `DequantizationError` with tensor name and reason.

### 6.2 Numerical Accuracy

Dequantization uses the exact inverse of the quantization formula:
- INT8/INT4: `dequantized = quantized.astype(float32) * scale`
- Binary: `dequantized = scale * (1 if bit else -1)`
- Ternary: `dequantized = scale * {-1, 0, +1}[trit]`

Output dtype configurable (default float16).

### 6.3 Validation Against Original

Runtime provides `validate_tensor(name, original_tensor)` returning:
- MAE, MSE, Max Abs Error, Relative Error
- Cosine similarity, SNR, PSNR
- Uses existing `ldmark.compression.metrics.calculate_error_metrics`

---

## 7. Computation Interface

### 7.1 Matrix Multiplication (Required)

```python
output = runtime.matmul(
    weight_name="layer.0.weight",
    input_tensor=input_array,  # [batch, in_features]
)
```

**Behavior:**
1. Loads and decompresses weight tensor (if not already)
2. Converts input to `target_dtype` (default float16)
3. Performs `np.matmul(input, weight.T)` or `np.matmul(input, weight)` based on shapes
4. Returns `MatmulResult` with output, timing, memory snapshot

**Shape Convention:**
- Weight: `[out_features, in_features]` (standard PyTorch convention)
- Input: `[batch, in_features]` or `[in_features]`
- Output: `[batch, out_features]` or `[out_features]`

### 7.2 Cached Weight Multiplication

```python
output = runtime.matmul_with_cached_weight(weight_name, input_tensor)
```

Assumes weight already loaded and decompressed. Use for repeated operations.

---

## 8. Unsupported Features

The following are **explicitly NOT implemented** in this runtime:

| Feature | Status | Notes |
|---------|--------|-------|
| CUDA/ROCm/Metal kernels | ❌ Not implemented | CPU/NumPy only |
| Direct low-bit computation | ❌ Not implemented | Always dequantizes to FP16/FP32 first |
| KV-cache management | ❌ Not implemented | Out of scope |
| Speculative decoding | ❌ Not implemented | Out of scope |
| Quantized attention | ❌ Not implemented | Out of scope |
| Batch processing > 1 | ⚠️ Limited | Works but not optimized |
| Multi-threaded loading | ❌ Not implemented | Single-threaded |
| Streaming / async I/O | ❌ Not implemented | Synchronous only |
| Model sharding | ❌ Not implemented | Single artifact only |
| Dynamic quantization | ❌ Not implemented | Static artifact only |

---

## 9. Future Kernel Interface

To enable future optimized kernels, the following extension points are defined:

### 9.1 Kernel Registry (Future)

```python
# Future: runtime.register_kernel(encoding, backend, kernel_fn)
# Future: runtime.set_backend("cuda" | "rocm" | "metal" | "cpu")
```

### 9.2 Required Kernel Interface

```python
def kernel_dequantize_int8(
    packed_data: np.ndarray,      # uint8
    scales: np.ndarray,           # float16
    original_shape: tuple,
    group_size: int,
    output: np.ndarray,           # Pre-allocated output (float16)
) -> None:
    """In-place dequantization kernel."""
    pass

def kernel_matmul_int8(
    input: np.ndarray,            # float16 [M, K]
    weight_packed: np.ndarray,    # uint8 [N, K] packed
    weight_scales: np.ndarray,    # float16 [N, K//group_size]
    output: np.ndarray,           # Pre-allocated output [M, N] float16
    group_size: int,
) -> None:
    """Direct INT8 matmul without full dequantization."""
    pass
```

### 9.3 Compatibility Requirements

Any future kernel implementation MUST:
1. Produce bit-identical results to NumPy reference (within float16 precision)
2. Accept same metadata from `TensorIndexEntry`
3. Raise `UnsupportedRuntimeFormat` for unsupported encodings
4. Maintain memory accounting interface

---

## 10. Error Handling Contract

| Exception | When Raised | Required Fields |
|-----------|-------------|-----------------|
| `UnsupportedRuntimeFormat` | Unsupported encoding/target_bits | `format_name`, `message` |
| `TensorNotFoundError` | Tensor not in manifest | `tensor_name`, `available_tensors` |
| `ArtifactCorruptedError` | Manifest missing, checksum fail, size mismatch | `message`, `tensor_name` (optional) |
| `DequantizationError` | Dequantization validation fails | `tensor_name`, `reason` |
| `ComputationError` | Matmul shape mismatch, etc. | `operation`, `reason` |

All exceptions inherit from `LDMARKRuntimeError`.

---

## 11. Versioning & Evolution

### 11.1 Artifact Version Compatibility

- Runtime v1.x reads artifact v1.x
- Runtime MUST reject artifact v2+ with clear error
- New artifact versions SHOULD be additive (new fields, not breaking changes)

### 11.2 Adding New Formats

To add a new compression format:
1. Add encoding to `TensorEncoding` enum in artifact format
2. Add dequantization function in `dequantize.py`
3. Add dispatch in `dequantize_tensor()`
4. Add tests in `tests/runtime/`
5. Update this CONTRACT.md

---

## 12. Reference Implementation

The reference implementation is in `src/ldmark/runtime/`:
- `runtime.py` — Main `LDMARKRuntime` class
- `dequantize.py` — Format-specific dequantization
- `memory.py` — Memory accounting
- `benchmark.py` — Benchmarking utilities
- `exceptions.py` — Custom exceptions

All implementations use NumPy only. No external dependencies beyond NumPy.

---

## 13. Conformance Testing

A conforming runtime MUST pass:
1. All tests in `tests/runtime/test_runtime.py`
2. Correctness tests for each supported format (INT8, INT4, binary, ternary)
3. Memory accounting accuracy tests
4. Error handling tests (missing tensor, corrupted artifact, unsupported format)
5. Benchmark reproducibility (same artifact → same timing within 10%)

---

## Appendix: Example Usage

```python
from ldmark.runtime import open_runtime

# Open artifact (lazy - no tensors loaded)
with open_runtime("model.ldmark") as runtime:
    # Inspect metadata
    print(f"Model: {runtime.get_model_info().model_id}")
    print(f"Tensors: {runtime.list_tensors()}")
    
    # Load and decompress single tensor
    weight = runtime.get_tensor("layer.0.weight", decompress=True)
    print(f"Shape: {weight.dequantized.data.shape}")
    print(f"Compressed: {weight.entry.byte_length} bytes")
    print(f"Decompressed: {weight.dequantized.decompressed_size_bytes} bytes")
    
    # Matrix multiplication
    input_data = np.random.randn(1, 4096).astype(np.float16)
    result = runtime.matmul("layer.0.weight", input_data)
    print(f"Output shape: {result.output.shape}")
    print(f"Compute time: {result.computation_time_ms:.2f} ms")
    
    # Memory accounting
    runtime.print_memory_summary()
```

---

**End of Contract**