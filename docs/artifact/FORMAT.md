# LDMARK Artifact Format Specification

**Version:** 1.0 (Experimental)  
**Format Identifier:** `ldmark-artifact-1.0`  
**Status:** Experimental - Subject to change

---

## Overview

The LDMARK artifact format is a versioned, deterministic container format for storing compiled neural network models. It is designed to be:

- **Deterministic**: Identical inputs produce identical artifacts
- **Versioned**: Explicit format versioning with forward-compatibility guarantees
- **Inspectable**: Metadata readable without loading tensor payloads
- **Portable**: Platform-independent directory structure
- **Extensible**: Designed for future evolution

The format does **not** implement inference, runtime kernels, or model execution. It is purely a storage and interchange format.

---

## Artifact Structure

An LDMARK artifact is a directory with the following structure:

```
model.ldmark/
├── manifest.json          # Primary manifest (required)
└── tensors/
    ├── <tensor_name>.bin          # Compressed tensor data
    └── <tensor_name>.scales.bin   # Scale factors (optional, per tensor)
```

### File Descriptions

| File | Required | Description |
|------|----------|-------------|
| `manifest.json` | Yes | Complete artifact metadata, tensor index, and integrity checksums |
| `tensors/<name>.bin` | Per tensor | Raw compressed tensor bytes |
| `tensors/<name>.scales.bin` | Optional | Scale factors for quantized tensors (float16) |

### Naming Convention

Tensor filenames are derived from tensor names by replacing `.`, `/`, and `\` with `_`. For example:
- `model.layers.0.self_attn.qkv_proj.weight` → `model_layers_0_self_attn_qkv_proj_weight.bin`

---

## Manifest Format (`manifest.json`)

The manifest is a JSON document containing all metadata needed to interpret the artifact without loading tensor data.

### Top-Level Structure

```json
{
  "magic": "LDMARK",
  "format_version": "ldmark-artifact-1.0",
  "model": { ... },
  "compression": { ... },
  "compilation": { ... },
  "hardware": { ... },
  "tensors": [ ... ],
  "validation": { ... },
  "storage": { ... },
  "manifest_sha256": "<hex>"
}
```

### Field Descriptions

#### Magic & Version
| Field | Type | Description |
|-------|------|-------------|
| `magic` | string | Always `"LDMARK"` for format identification |
| `format_version` | string | Format version identifier (e.g., `"ldmark-artifact-1.0"`) |

#### Model Information (`model`)
```json
{
  "model_id": "string",
  "architecture": "string",
  "parameter_count": "integer",
  "tensor_count": "integer",
  "num_layers": "integer|null",
  "hidden_size": "integer|null",
  "intermediate_size": "integer|null",
  "num_attention_heads": "integer|null",
  "num_kv_heads": "integer|null",
  "vocab_size": "integer|null",
  "max_context_length": "integer|null",
  "original_dtype": "string",
  "source_path": "string"
}
```

#### Compression Configuration (`compression`)
```json
{
  "method": "fp16|int8|int4|binary|ternary|prismml_q1_0_g128",
  "target_bits": "integer",
  "group_size": "integer",
  "scale_dtype": "string",
  "symmetric": "boolean",
  "per_tensor_overrides": { "tensor_name": { ...quantization params... } }
}
```

#### Compilation Metadata (`compilation`)
```json
{
  "ldmark_version": "string",
  "format_version": "string",
  "created_at": "ISO8601 timestamp",
  "optimization_strategy": "size|speed|balanced|accuracy",
  "target_precision": "string",
  "max_model_size_gb": "float|null",
  "config_snapshot": { ... },
  "warnings": [ "string", ... ]
}
```

#### Hardware Target (`hardware`)
```json
{
  "cpu_model": "string",
  "cpu_cores": "integer",
  "cpu_architecture": "string",
  "system_ram_gb": "float",
  "gpu_model": "string|null",
  "gpu_vram_gb": "float|null",
  "gpu_vendor": "string|null",
  "operating_system": "string",
  "python_version": "string"
}
```

#### Tensor Index (`tensors`)
Array of tensor index entries (see [Tensor Index Entry](#tensor-index-entry) below).

#### Validation Results (`validation`)
```json
{
  "status": "passed|failed|skipped|not_run",
  "passed": "boolean",
  "details": { ... },
  "error": "string|null"
}
```

#### Storage Accounting (`storage`)
```json
{
  "raw_tensor_bytes": "integer",
  "scale_bytes": "integer",
  "metadata_bytes": "integer",
  "manifest_bytes": "integer",
  "total_artifact_bytes": "integer",
  "theoretical_bits_per_weight": "float",
  "actual_bits_per_weight": "float",
  "parameter_count": "integer"
}
```

#### Manifest Checksum
| Field | Type | Description |
|-------|------|-------------|
| `manifest_sha256` | string | SHA-256 of manifest JSON (excluding this field) |

---

## Tensor Index Entry

Each tensor in the artifact has an entry in the `tensors` array containing all metadata needed to locate and interpret the tensor.

```json
{
  "name": "model.layers.0.self_attn.qkv_proj.weight",
  "original_shape": [384, 128],
  "original_dtype": "float16",
  "original_size_bytes": 98304,
  "encoding": "groupwise_quantized",
  "quantization": {
    "target_bits": 4,
    "group_size": 128,
    "scale_dtype": "float16",
    "symmetric": true,
    "zero_point": false
  },
  "filename": "model_layers_0_self_attn_qkv_proj_weight.bin",
  "offset": 0,
  "byte_length": 24576,
  "scale_filename": "model_layers_0_self_attn_qkv_proj_weight.scales.bin",
  "scale_offset": 0,
  "scale_byte_length": 512,
  "num_scales": 256,
  "sha256": "<hex>",
  "scale_sha256": "<hex>",
  "mae": 0.002,
  "mse": 0.000005,
  "max_abs_error": 0.0068,
  "relative_error": 0.117,
  "bits_per_weight": 4.125
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Original tensor name |
| `original_shape` | int[] | Yes | Shape before compression |
| `original_dtype` | string | Yes | NumPy dtype string (e.g., `"float16"`) |
| `original_size_bytes` | int | Yes | Uncompressed size in bytes |
| `encoding` | string | Yes | `"raw"`, `"groupwise_quantized"`, `"binary_packed"`, `"ternary_packed"` |
| `quantization` | object | For quantized | Quantization parameters (see below) |
| `filename` | string | Yes | Tensor data filename in `tensors/` |
| `offset` | int | Yes | Byte offset within file (for future multi-tensor files) |
| `byte_length` | int | Yes | Compressed tensor data size in bytes |
| `scale_filename` | string | No | Scale data filename (if separate) |
| `scale_offset` | int | No | Byte offset in scale file |
| `scale_byte_length` | int | No | Scale data size in bytes |
| `num_scales` | int | No | Number of scale factors |
| `sha256` | string | No | SHA-256 of tensor data |
| `scale_sha256` | string | No | SHA-256 of scale data |
| `mae` | float | No | Mean absolute error from validation |
| `mse` | float | No | Mean squared error from validation |
| `max_abs_error` | float | No | Maximum absolute error |
| `relative_error` | float | No | Relative error |
| `bits_per_weight` | float | No | Actual bits per weight including overhead |

### Quantization Parameters

```json
{
  "target_bits": 4,
  "group_size": 128,
  "scale_dtype": "float16",
  "symmetric": true,
  "zero_point": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `target_bits` | int | Target bit width (1, 2, 4, 8, 16) |
| `group_size` | int | Number of weights per scale group |
| `scale_dtype` | string | NumPy dtype for scale factors |
| `symmetric` | bool | Whether quantization is symmetric |
| `zero_point` | bool | Whether zero-point is used |

---

## Tensor Data Format

### Groupwise Quantized (`encoding: "groupwise_quantized"`)

For INT8/INT4 quantization with group-wise scaling:

**Tensor Data File (`.bin`):**
- INT8: Raw int8 bytes, concatenated groups
- INT4: Packed uint8 bytes (2 weights per byte, high nibble first)

**Scale Data File (`.scales.bin`):**
- Raw float16 values, one per group, concatenated

### Binary Packed (`encoding: "binary_packed"`)

For binary (-1, +1) quantization:

**Tensor Data File (`.bin`):**
- Packed bits: 1 bit per weight (0 = -1, 1 = +1), LSB first within byte

**Scale Data File (`.scales.bin`):**
- Raw float16 values, one per group

### Ternary Packed (`encoding: "ternary_packed"`)

For ternary (-1, 0, +1) quantization:

**Tensor Data File (`.bin`):**
- Packed 2-bit values per weight: 00=-1, 01=0, 10=+1

**Scale Data File (`.scales.bin`):**
- Raw float16 values, one per group

---

## Integrity Verification

### Manifest Checksum
The `manifest_sha256` field contains a SHA-256 hash of the manifest JSON (with `manifest_sha256` field removed, keys sorted, no whitespace). This allows detection of manifest tampering.

### Tensor Checksums
Each tensor entry may contain:
- `sha256`: SHA-256 of the tensor data file
- `scale_sha256`: SHA-256 of the scale data file

### Size Verification
Each tensor entry includes `byte_length` and `scale_byte_length` for size verification.

### Verification Procedure
1. Read and parse `manifest.json`
2. Verify `manifest_sha256` matches computed hash
3. For each tensor:
   - Verify file exists
   - Verify `byte_length` matches file size
   - If `sha256` present, verify checksum
   - If scale file present, verify scale file size and checksum

---

## Storage Accounting

The artifact tracks detailed storage breakdown:

| Component | Description |
|-----------|-------------|
| `raw_tensor_bytes` | Sum of all tensor data file sizes |
| `scale_bytes` | Sum of all scale file sizes |
| `metadata_bytes` | Estimated metadata overhead |
| `manifest_bytes` | Size of manifest.json |
| `total_artifact_bytes` | Total disk usage |
| `theoretical_bits_per_weight` | Expected bits/weight from compression config |
| `actual_bits_per_weight` | `total_artifact_bytes * 8 / parameter_count` |
| `parameter_count` | Total model parameters |

**Important**: Theoretical compression (e.g., 4.125 bits/weight for INT4) differs from actual artifact size due to metadata, scale factors, file system overhead, and packing inefficiencies.

---

## Versioning

### Format Version
Format versions follow the pattern: `ldmark-artifact-MAJOR.MINOR`

- **MAJOR**: Breaking changes (incompatible structure)
- **MINOR**: Backward-compatible additions

### Compatibility Rules
1. **Readers MUST reject** artifacts with unsupported MAJOR version
2. **Readers SHOULD accept** artifacts with newer MINOR version (ignore unknown fields)
3. **Writers MUST produce** the latest format version they support
4. **Artifacts MUST include** `format_version` in manifest

### Current Version
- **Latest**: `ldmark-artifact-1.0`
- **Minimum Supported**: `ldmark-artifact-1.0`

---

## Usage Examples

### Python API - Writing an Artifact

```python
from ldmark.artifact import (
    LDMARKArtifactWriter,
    ModelInfo,
    CompressionInfo,
    CompilationInfo,
    HardwareInfo,
    ValidationInfo,
    QuantizationParams,
    CompressionMethod,
    ArtifactFormatVersion,
)
import numpy as np

writer = LDMARKArtifactWriter("model.ldmark", overwrite=True)

writer.set_model_info(ModelInfo(
    model_id="my_model",
    architecture="llama",
    parameter_count=787072,
    tensor_count=17,
    num_layers=2,
    hidden_size=128,
    vocab_size=1024,
    source_path="/path/to/original.npz",
))

writer.set_compression_info(CompressionInfo(
    method=CompressionMethod.INT4,
    target_bits=4,
    group_size=128,
    scale_dtype="float16",
    symmetric=True,
))

writer.set_compilation_info(CompilationInfo(
    ldmark_version="0.1.0",
    format_version="ldmark-artifact-1.0",
    created_at="2024-01-15T10:30:00",
    optimization_strategy="balanced",
))

# Add tensors
for name, data, scales in tensor_data:
    writer.add_tensor(
        name=name,
        data=data,
        scales=scales,
        original_shape=original_shape,
        original_dtype=np.float16,
        quantization=QuantizationParams(target_bits=4, group_size=128),
        mae=mae, mse=mse, max_abs_error=max_err, relative_error=rel_err,
    )

artifact = writer.write()
```

### Python API - Reading an Artifact

```python
from ldmark.artifact import open_artifact, inspect_artifact

# Metadata-only inspection (fast, no tensor loading)
info = inspect_artifact("model.ldmark")
print(f"Model: {info['model']['model_id']}")
print(f"Parameters: {info['model']['parameter_count']:,}")
print(f"Compression: {info['compression']['method']}")

# Full access with lazy tensor loading
with open_artifact("model.ldmark") as reader:
    for name in reader.list_tensors():
        tensor = reader.read_tensor(name)
        print(f"{name}: {tensor.data.shape} compressed, {tensor.scales.shape} scales")

# Verify integrity
from ldmark.artifact import verify_artifact
report = verify_artifact("model.ldmark")
if report.all_valid:
    print("Artifact integrity verified!")
else:
    print(f"Integrity issues: {report.errors}")
```

### CLI Inspection

```bash
# Using Python to inspect
python -c "
from ldmark.artifact import inspect_artifact
import json
info = inspect_artifact('model.ldmark')
print(json.dumps(info, indent=2))
"
```

---

## Extensibility

The format is designed for future extensions:

1. **Unknown fields in manifest** are ignored by readers (forward compatibility)
2. **Per-tensor overrides** in compression config allow mixed precision
3. **New encodings** can be added to `TensorEncoding` enum
4. **Custom metadata** can be added to `config_snapshot` in compilation info
5. **Multi-file tensor storage** supported via `offset`/`byte_length`

---

## Limitations (Experimental)

- **No single-file container**: Currently directory-based only
- **No encryption**: Artifacts are not encrypted
- **No compression of manifest**: manifest.json is stored as plain JSON
- **No streaming writes**: All tensor data must be available in memory during write
- **No incremental updates**: Artifacts are written atomically

---

## Migration from `ldmark-compilation-0.1`

The previous `CompilationArtifact` format (`ldmark-compilation-0.1`) was an intermediate format. Key differences:

| Aspect | `ldmark-compilation-0.1` | `ldmark-artifact-1.0` |
|--------|-------------------------|----------------------|
| Structure | Directory + metadata.json + tensors/ | Directory + manifest.json + tensors/ |
| Tensor storage | `.npz` with data+scales+shape | Separate `.bin` + `.scales.bin` |
| Manifest | Implicit (metadata.json) | Explicit (manifest.json) |
| Checksums | None | SHA-256 for manifest + tensors |
| Storage accounting | Basic | Detailed breakdown |
| Versioning | None | Explicit format_version |

Migration requires re-compilation. The compiler's `ExportStage` will be updated to produce the new format.

---

## Future Considerations

1. **Single-file container**: TAR-based or custom binary container for distribution
2. **Memory-mapped reads**: Already supported via `mmap_tensors` option
3. **Parallel tensor loading**: For multi-threaded inference
4. **Delta updates**: For fine-tuned models
5. **Compression of metadata**: gzip/zstd for large manifests
6. **Signed artifacts**: Cryptographic signing for supply chain security