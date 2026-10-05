"""
LDMARK Runtime - Example Usage

Demonstrates how to use the LDMARK runtime with a synthetic artifact.
"""

from __future__ import annotations

import tempfile
import numpy as np
from pathlib import Path

from src.ldmark.artifact.writer import (
    LDMARKArtifactWriter,
    create_artifact_from_pipeline,
)
from src.ldmark.artifact.format import (
    ArtifactFormatVersion,
    ModelInfo,
    CompressionInfo,
    HardwareInfo,
    TensorEncoding,
    QuantizationParams,
    CompressionMethod,
)
from src.ldmark.compression.quantize import quantize_int8, quantize_int4, QuantizationConfig, QuantizationTarget
from src.ldmark.compression.binary_ternary import binary_quantize_sign, ternary_quantize_threshold, BinaryConfig, TernaryConfig
from src.ldmark.runtime import open_runtime, LDMARKRuntime


def create_synthetic_artifact(artifact_dir: str) -> None:
    """Create a synthetic LDMARK artifact with multiple compression formats for testing."""
    
    writer = LDMARKArtifactWriter(artifact_dir, ArtifactFormatVersion.V1, overwrite=True)
    
    # Model info
    model_info = ModelInfo(
        model_id="synthetic-test-model",
        architecture="llama",
        parameter_count=1_000_000,
        tensor_count=4,
        num_layers=2,
        hidden_size=512,
        intermediate_size=2048,
        num_attention_heads=8,
        num_kv_heads=8,
        vocab_size=1000,
        max_context_length=2048,
        original_dtype="float16",
    )
    writer.set_model_info(model_info)
    
    # Hardware info
    hardware_info = HardwareInfo(
        cpu_model="Test CPU",
        cpu_cores=8,
        cpu_architecture="x86_64",
        system_ram_gb=16.0,
        operating_system="Linux",
        python_version="3.10",
    )
    writer.set_hardware_info(hardware_info)
    
    # Create test tensors with different compression formats
    
    # Tensor 1: INT8 quantized
    tensor1 = np.random.randn(512, 1024).astype(np.float32) * 0.1
    qconfig8 = QuantizationConfig(target_bits=QuantizationTarget.INT8, group_size=128)
    qtensor8 = quantize_int8(tensor1, group_size=128)
    
    writer.add_tensor(
        name="layer.0.weight",
        data=qtensor8.data,
        scales=qtensor8.scales,
        original_shape=tensor1.shape,
        original_dtype=tensor1.dtype,
        encoding=TensorEncoding.GROUPWISE_QUANTIZED,
        quantization=QuantizationParams(
            target_bits=8,
            group_size=128,
            scale_dtype="float16",
            symmetric=True,
        ),
    )
    
    # Tensor 2: INT4 quantized
    tensor2 = np.random.randn(512, 2048).astype(np.float32) * 0.1
    qtensor4 = quantize_int4(tensor2, group_size=128)
    
    writer.add_tensor(
        name="layer.1.weight",
        data=qtensor4.data,
        scales=qtensor4.scales,
        original_shape=tensor2.shape,
        original_dtype=tensor2.dtype,
        encoding=TensorEncoding.GROUPWISE_QUANTIZED,
        quantization=QuantizationParams(
            target_bits=4,
            group_size=128,
            scale_dtype="float16",
            symmetric=True,
        ),
    )
    
    # Tensor 3: Binary quantized
    tensor3 = np.random.randn(256, 512).astype(np.float32) * 0.1
    bconfig = BinaryConfig(group_size=128)
    btensor = binary_quantize_sign(tensor3, bconfig)
    
    writer.add_tensor(
        name="layer.0.attention.q_proj",
        data=btensor.packed_data,
        scales=btensor.scales,
        original_shape=tensor3.shape,
        original_dtype=tensor3.dtype,
        encoding=TensorEncoding.BINARY_PACKED,
        quantization=QuantizationParams(
            target_bits=1,
            group_size=128,
            scale_dtype="float16",
            symmetric=True,
        ),
    )
    
    # Tensor 4: Ternary quantized
    tensor4 = np.random.randn(256, 512).astype(np.float32) * 0.1
    tconfig = TernaryConfig(group_size=128)
    ttensor = ternary_quantize_threshold(tensor4, tconfig, threshold=0.05)
    
    writer.add_tensor(
        name="layer.1.attention.k_proj",
        data=ttensor.packed_data,
        scales=ttensor.scales,
        original_shape=tensor4.shape,
        original_dtype=tensor4.dtype,
        encoding=TensorEncoding.TERNARY_PACKED,
        quantization=QuantizationParams(
            target_bits=2,
            group_size=128,
            scale_dtype="float16",
            symmetric=True,
        ),
    )
    
    # Write artifact
    artifact = writer.write()
    print(f"Created artifact at {artifact_dir}")
    print(f"  Model: {artifact.manifest.model.model_id}")
    print(f"  Tensors: {len(artifact.manifest.tensors)}")
    print(f"  Total compressed: {artifact.manifest.get_total_compressed_bytes() / 1024:.1f} KB")


def demo_runtime(artifact_dir: str) -> None:
    """Demonstrate runtime usage."""
    
    print("\n" + "=" * 60)
    print("LDMARK RUNTIME DEMO")
    print("=" * 60)
    
    # Open runtime (lazy - no tensors loaded yet)
    with open_runtime(artifact_dir, target_dtype=np.float16) as runtime:
        # Inspect metadata
        print(f"\nFormat version: {runtime.get_format_version().value}")
        model = runtime.get_model_info()
        print(f"Model: {model.model_id} ({model.parameter_count:,} params, {model.tensor_count} tensors)")
        print(f"Tensors: {runtime.list_tensors()}")
        
        # Load and decompress a single tensor (lazy loading)
        print("\n--- Loading INT8 tensor ---")
        tensor = runtime.get_tensor("layer.0.weight", decompress=True)
        print(f"  Name: {tensor.name}")
        print(f"  Shape: {tensor.dequantized.data.shape}")
        print(f"  Dtype: {tensor.dequantized.data.dtype}")
        print(f"  Compressed: {tensor.dequantized.compressed_size_bytes:,} bytes")
        print(f"  Decompressed: {tensor.dequantized.decompressed_size_bytes:,} bytes")
        print(f"  Encoding: {tensor.dequantized.encoding.value}")
        
        # Matrix multiplication
        print("\n--- Matrix Multiplication ---")
        input_data = np.random.randn(1, 1024).astype(np.float16)
        result = runtime.matmul("layer.0.weight", input_data)
        print(f"  Input shape: {result.input_shape}")
        print(f"  Output shape: {result.output_shape}")
        print(f"  Compute time: {result.computation_time_ms:.2f} ms")
        
        # Memory accounting
        print("\n--- Memory Accounting ---")
        runtime.print_memory_summary()
        
        # Test other formats
        print("\n--- Testing Other Formats ---")
        for name in ["layer.1.weight", "layer.0.attention.q_proj", "layer.1.attention.k_proj"]:
            t = runtime.get_tensor(name, decompress=True)
            print(f"  {name}: {t.dequantized.data.shape}, "
                  f"encoding={t.dequantized.encoding.value}, "
                  f"compressed={t.dequantized.compressed_size_bytes} bytes, "
                  f"decompressed={t.dequantized.decompressed_size_bytes} bytes")
        
        # Demonstrate lazy loading - release and reload
        print("\n--- Lazy Loading Demo ---")
        runtime.release_tensor("layer.0.weight")
        print("Released layer.0.weight")
        runtime.print_memory_summary()
        
        # Reload
        tensor = runtime.get_tensor("layer.0.weight", decompress=True)
        print("Reloaded layer.0.weight")
        runtime.print_memory_summary()


def demo_benchmark(artifact_dir: str) -> None:
    """Run benchmark demo."""
    
    print("\n" + "=" * 60)
    print("BENCHMARK DEMO")
    print("=" * 60)
    
    with open_runtime(artifact_dir, target_dtype=np.float16) as runtime:
        # Benchmark a single tensor
        result = runtime.benchmark_tensor(
            "layer.0.weight",
            input_shape=(1, 1024),
            num_warmup=2,
            num_runs=5,
        )
        
        from src.ldmark.runtime.benchmark import print_benchmark_result
        print_benchmark_result(result)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir) / "test_artifact"
        
        # Create synthetic artifact
        create_synthetic_artifact(str(artifact_dir))
        
        # Demo runtime
        demo_runtime(str(artifact_dir))
        
        # Demo benchmark
        demo_benchmark(str(artifact_dir))
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()