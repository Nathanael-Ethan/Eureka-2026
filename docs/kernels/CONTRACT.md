# LDMARK Kernel Contract

**Version:** 1.0 (Experimental)  
**Status:** Research baseline for future optimized backends

---

## Overview

This document defines the interface contract for LDMARK low-bit computation kernels. The contract separates **weight representation** from **computation backend**, enabling multiple implementations (NumPy reference, CUDA, ROCm, Metal, etc.) to work with the same compressed weight formats.

The contract covers four weight representations:
- **INT8**: 8-bit symmetric group-wise quantization
- **INT4**: 4-bit symmetric group-wise quantization (packed)
- **Binary**: 1-bit sign weights (packed)
- **Ternary**: 2-bit weights {-1, 0, +1} (packed)

---

## Core Abstractions

### 1. LowBitMatrix (Weight Representation)

Abstract base class for compressed weight matrices.

```python
class LowBitMatrix(ABC):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:  # (out_features, in_features)
        pass
    
    @property
    @abstractmethod
    def dtype(self) -> np.dtype:  # Storage dtype
        pass
    
    @property
    @abstractmethod
    def compressed_size_bytes(self) -> int:
        pass
    
    @property
    @abstractmethod
    def decompressed_size_bytes(self) -> int:
        pass
    
    @abstractmethod
    def dequantize(self) -> np.ndarray:
        """Full dequantization to FP32 for reference/validation."""
        pass
    
    @abstractmethod
    def matmul(self, x: np.ndarray, config: ComputationConfig) -> ComputeResult:
        """Direct matrix multiplication Y = W @ X."""
        pass
```

**Implementations:**
- `Int8Matrix` - INT8 group-wise quantized
- `Int4Matrix` - INT4 packed group-wise quantized  
- `BinaryMatrix` - Binary sign weights (packed bits)
- `TernaryMatrix` - Ternary weights (packed 2-bit trits)

---

### 2. ComputationConfig (Computation Parameters)

```python
@dataclass(frozen=True)
class ComputationConfig:
    output_precision: ComputePrecision = ComputePrecision.FP32
    accumulate_in_fp32: bool = True   # Use FP32 accumulation for numerical stability
    batch_size: int = 1
    enable_memory_tracking: bool = True
```

---

### 3. ComputePrecision (Output Precision)

```python
class ComputePrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
```

---

### 4. LowBitMatmul (Backend Interface)

Abstract interface for computation backends.

```python
class LowBitMatmul(ABC):
    @abstractmethod
    def compute(
        self,
        weight: LowBitMatrix,
        x: np.ndarray,
        config: ComputationConfig
    ) -> ComputeResult:
        pass
    
    @property
    @abstractmethod
    def supported_precisions(self) -> List[ComputePrecision]:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

**Reference Implementations:**
- `Int8MatmulHorizontal` - INT8 horizontal grouping
- `Int4MatmulHorizontal` - INT4 horizontal grouping
- `BinaryMatmulHorizontal` - Binary horizontal grouping
- `TernaryMatmulHorizontal` - Ternary horizontal grouping

---

### 5. ComputeResult (Computation Output)

```python
@dataclass
class ComputeResult:
    output: np.ndarray
    memory_stats: MemoryStats
    timing_stats: TimingStats
    metadata: Dict[str, Any]
```

---

### 6. MemoryStats (Memory Accounting)

```python
@dataclass
class MemoryStats:
    compressed_weight_bytes: int = 0
    decompressed_weight_bytes: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    temporary_bytes: int = 0
    peak_bytes: int = 0
```

---

### 7. TimingStats (Performance Metrics)

```python
@dataclass
class TimingStats:
    preparation_time_s: float = 0.0
    computation_time_s: float = 0.0
    total_time_s: float = 0.0
```

---

## Weight Representations

### INT8 (Group-wise Quantization)

**Storage:** Int8 values + FP16 scales per group
**Grouping:** Horizontal (per-row), `group_size` columns per group
**Formula:** `W ≈ scale_g × Q_g` where `Q_g ∈ [-128, 127]`

**Quantization Config:**
```python
@dataclass(frozen=True)
class QuantizationConfig:
    target_bits: QuantizationTarget = QuantizationTarget.INT8
    group_size: int = 128
    scale_dtype: np.dtype = np.float16
    symmetric: bool = True
```

**Dequantization:** `W[i, k] = Q[i, k] × scale_g` where `g = (i * cols + k) // group_size`

---

### INT4 (Packed Group-wise Quantization)

**Storage:** Packed uint8 (2 nibbles/byte) + FP16 scales per group
**Grouping:** Horizontal (per-row), `group_size` columns per group
**Formula:** `W ≈ scale_g × Q_g` where `Q_g ∈ [-8, 7]`

**Packing:** Two INT4 values per byte (high nibble first, sign-extended)

**Dequantization:** Unpack nibbles, then `W[i, k] = Q[i, k] × scale_g`

---

### Binary (Packed Sign Weights)

**Storage:** Packed bits (1 bit/weight) + FP16 scales per group
**Grouping:** Horizontal (per-row)
**Values:** `{0 → -1, 1 → +1}`

**Packing:** 8 weights per byte (LSB first)

**Dequantization:** Unpack bits, map `0→-1, 1→+1`, then `W[i, k] = sign[i, k] × scale_g`

---

### Ternary (Packed 2-bit Trits)

**Storage:** Packed 2-bit trits (4 trits/byte) + FP16 scales per group
**Grouping:** Horizontal (per-row)
**Values:** `{00 → -1, 01 → 0, 10 → +1}`

**Packing:** 4 trits per byte (2 bits each)

**Dequantization:** Unpack trits, map `{00→-1, 01→0, 10→+1}`, then `W[i, k] = trit[i, k] × scale_g`

---

## Computation Algorithm (Horizontal Grouping)

For all representations with horizontal grouping:

```
Y = W @ X  where W ∈ ℝ^(rows × cols), X ∈ ℝ^(cols × N)

For each output row r:
    For each group g in row r:
        k_start = g × group_size
        k_end = min(k_start + group_size, K)
        
        # Extract weight slice for this row's group
        w_group = unpacked_weight[r, k_start:k_end]  # shape (group_k,)
        
        # Extract input slice
        x_group = X[k_start:k_end, :]  # shape (group_k, N)
        
        # Compute partial: (1, group_k) @ (group_k, N) → (1, N)
        partial = w_group @ x_group
        
        # Accumulate with group scale
        Y[r, :] += scale_g × partial
```

**Key Properties:**
- No full weight matrix dequantization
- Only unpacks weight slices needed for current group
- Accumulates in FP32 for numerical stability (configurable)
- Falls back to full dequantization for non-horizontal group layouts

---

## Backend Interface Requirements

Any optimized backend (CUDA, ROCm, Metal, etc.) must implement:

### 1. Weight Representation Classes

For each representation, provide a class implementing `LowBitMatrix`:
- Construction from quantized tensor data
- `shape`, `dtype`, `compressed_size_bytes`, `decompressed_size_bytes`
- `dequantize()` → FP32 numpy array
- `matmul(x, config)` → `ComputeResult`

### 2. Computation Kernel

For each representation, provide a class implementing `LowBitMatmul`:
- `compute(weight, x, config)` → `ComputeResult`
- `supported_precisions` property
- `name` property

### 3. Numerical Correctness

All backends must pass the numerical correctness tests:
- Match reference (dequantize-then-matmul) within tolerance
- Support all test shapes: square, rectangular, odd dims, non-divisible groups
- Deterministic output for same inputs

### 4. Memory Accounting

Backends must report accurate `MemoryStats`:
- `compressed_weight_bytes`: Actual storage size
- `decompressed_weight_bytes`: Size if fully dequantized to FP32
- `temporary_bytes`: Peak temporary allocation during computation
- `peak_bytes`: Total peak memory

### 4. Timing Reporting

Backends must report accurate `TimingStats`:
- `preparation_time_s`: Kernel launch / data transfer overhead
- `computation_time_s`: Actual kernel execution
- `total_time_s`: Sum of above

---

## Horizontal Grouping Validity

The horizontal grouping optimization applies when:

```python
horizontal_valid = (
    cols >= group_size and
    cols % group_size == 0 and
    n_groups == rows * (cols // group_size)
)
```

When `horizontal_valid` is False, the kernel **must** fall back to full dequantization then matmul to ensure correctness.

---

## Validation & Testing

### Reference Implementation

```python
def create_reference_result(weight: LowBitMatrix, x: np.ndarray, config: ComputationConfig):
    """Dequantize then matmul - the correctness baseline."""
    w_fp32 = weight.dequantize()
    output = w_fp32 @ x
    # ... package as ComputeResult
```

### Comparison Metrics

```python
def compare_results(direct: ComputeResult, reference: ComputeResult, rtol=1e-4, atol=1e-5):
    abs_error = np.abs(direct.output - reference.output)
    rel_error = abs_error / (np.abs(reference.output) + 1e-12)
    
    return {
        "passed": np.allclose(direct.output, reference.output, rtol=rtol, atol=atol),
        "max_absolute_error": float(np.max(abs_error)),
        "mean_absolute_error": float(np.mean(abs_error)),
        "max_relative_error": float(np.max(rel_error)),
        "mean_relative_error": float(np.mean(rel_error)),
    }
```

### Required Test Coverage

All backends must pass tests for:
- Square matrices (64×64, 128×128)
- Rectangular matrices (128×64, 64×128)
- Vector input (K, 1)
- Odd dimensions (33×31)
- Non-divisible group sizes (50×50, group=17)
- Single row (1×64)
- Single column (64×1)
- FP16 and FP32 output precision
- FP32 vs non-FP32 accumulation
- Deterministic output

---

## Future Extensions

### 1. Asymmetric Quantization
Add `zero_point` support to `QuantizationConfig`.

### 2. Mixed Precision
Per-group or per-row precision selection.

### 3. Sparse Representations
Support for structured sparsity + quantization.

### 4. Fused Operations
Bias add, activation, residual in same kernel.

### 5. Batch Processing
Native batched GEMM for transformer attention.

---

## Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026 | Initial contract for INT8/INT4/Binary/Ternary |

---

## Compliance

A backend is **contract-compliant** if:
1. Implements all `LowBitMatrix` and `LowBitMatmul` abstract methods
2. Passes all numerical correctness tests (rtol=1e-4, atol=1e-5)
2. Reports accurate `MemoryStats` and `TimingStats`
3. Falls back correctly for non-horizontal group layouts
4. Provides deterministic output for identical inputs

---

*This contract is experimental and subject to change based on backend implementation feedback.*