"""
LDMARK Artifact Writer

Creates LDMARK artifact directories with manifest, metadata, and tensor payloads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .format import (
    ArtifactFormat,
    ArtifactFormatVersion,
    Manifest,
    TensorIndexEntry,
    CompressionInfo,
    ModelInfo,
    HardwareInfo,
    CompilationInfo,
    ValidationInfo,
    StorageAccounting,
    TensorEncoding,
    QuantizationParams,
    CompressionMethod,
    create_compilation_info,
)
from .integrity import compute_sha256_bytes


@dataclass
class TensorPayload:
    """Container for tensor data and metadata during writing."""
    name: str
    data: np.ndarray
    scales: Optional[np.ndarray] = None
    original_shape: Tuple[int, ...] = ()
    original_dtype: np.dtype = np.float16
    encoding: TensorEncoding = TensorEncoding.GROUPWISE_QUANTIZED
    quantization: Optional[QuantizationParams] = None
    mae: float = 0.0
    mse: float = 0.0
    max_abs_error: float = 0.0
    relative_error: float = 0.0


class LDMARKArtifactWriter:
    """
    Writer for LDMARK compiled model artifacts.
    
    Creates a deterministic, versioned artifact directory structure:
    
    artifact_dir/
        manifest.json          # Main manifest with tensor index
        tensors/
            <tensor_name>.bin  # Compressed tensor data
            <tensor_name>.scales.bin  # Scale data (if separate)
    
    The format is designed to be:
    - Deterministic: Same inputs produce identical outputs
    - Inspectable: Manifest can be read without loading tensors
    - Portable: No platform-specific dependencies
    - Extensible: Versioned format with forward compatibility
    """
    
    def __init__(
        self,
        artifact_dir: Union[str, Path],
        format_version: ArtifactFormatVersion = ArtifactFormatVersion.V1,
        overwrite: bool = False,
    ):
        """
        Initialize writer.
        
        Args:
            artifact_dir: Output directory path
            format_version: Artifact format version to use
            overwrite: Whether to overwrite existing directory
        """
        self.artifact_dir = Path(artifact_dir).resolve()
        self.format_version = format_version
        self.overwrite = overwrite
        
        self.tensors_dir = self.artifact_dir / "tensors"
        self.manifest_path = self.artifact_dir / "manifest.json"
        
        # State
        self._tensor_entries: List[TensorIndexEntry] = []
        self._tensor_payloads: List[TensorPayload] = []
        self._model_info: Optional[ModelInfo] = None
        self._compression_info: Optional[CompressionInfo] = None
        self._hardware_info: Optional[HardwareInfo] = None
        self._compilation_info: Optional[CompilationInfo] = None
        self._validation_info: Optional[ValidationInfo] = None
        self._storage_accounting: Optional[StorageAccounting] = None
        self._warnings: List[str] = []
    
    def set_model_info(self, model_info: ModelInfo) -> "LDMARKArtifactWriter":
        """Set model identification information."""
        self._model_info = model_info
        return self
    
    def set_compression_info(self, compression_info: CompressionInfo) -> "LDMARKArtifactWriter":
        """Set global compression configuration."""
        self._compression_info = compression_info
        return self
    
    def set_hardware_info(self, hardware_info: HardwareInfo) -> "LDMARKArtifactWriter":
        """Set target hardware information."""
        self._hardware_info = hardware_info
        return self
    
    def set_compilation_info(self, compilation_info: CompilationInfo) -> "LDMARKArtifactWriter":
        """Set compilation metadata."""
        self._compilation_info = compilation_info
        return self
    
    def set_validation_info(self, validation_info: ValidationInfo) -> "LDMARKArtifactWriter":
        """Set validation results."""
        self._validation_info = validation_info
        return self
    
    def add_warning(self, warning: str) -> "LDMARKArtifactWriter":
        """Add a warning to the artifact."""
        self._warnings.append(warning)
        return self
    
    def add_tensor(
        self,
        name: str,
        data: np.ndarray,
        scales: Optional[np.ndarray] = None,
        original_shape: Optional[Tuple[int, ...]] = None,
        original_dtype: np.dtype = np.float16,
        encoding: TensorEncoding = TensorEncoding.GROUPWISE_QUANTIZED,
        quantization: Optional[QuantizationParams] = None,
        mae: float = 0.0,
        mse: float = 0.0,
        max_abs_error: float = 0.0,
        relative_error: float = 0.0,
    ) -> "LDMARKArtifactWriter":
        """
        Add a tensor to the artifact.
        
        Args:
            name: Tensor name (will be sanitized for filename)
            data: Compressed tensor data (numpy array)
            scales: Scale factors (numpy array, optional)
            original_shape: Original tensor shape
            original_dtype: Original tensor dtype
            encoding: How the tensor is encoded
            quantization: Quantization parameters
            mae: Mean absolute error from validation
            mse: Mean squared error from validation
            max_abs_error: Maximum absolute error
            relative_error: Relative error
        """
        payload = TensorPayload(
            name=name,
            data=data,
            scales=scales,
            original_shape=original_shape or data.shape,
            original_dtype=original_dtype,
            encoding=encoding,
            quantization=quantization,
            mae=mae,
            mse=mse,
            max_abs_error=max_abs_error,
            relative_error=relative_error,
        )
        self._tensor_payloads.append(payload)
        return self
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize tensor name for use as filename."""
        return name.replace(".", "_").replace("/", "_").replace("\\", "_")
    
    def _write_tensor_payload(self, payload: TensorPayload, index: int) -> TensorIndexEntry:
        """Write a single tensor payload to disk and return its index entry."""
        safe_name = self._sanitize_name(payload.name)
        
        # Determine filenames
        if payload.scales is not None and payload.scales.size > 0:
            # Separate scale file
            data_filename = f"{safe_name}.bin"
            scale_filename = f"{safe_name}.scales.bin"
        else:
            data_filename = f"{safe_name}.bin"
            scale_filename = ""
        
        data_path = self.tensors_dir / data_filename
        scale_path = self.tensors_dir / scale_filename if scale_filename else None
        
        # Write tensor data
        data_bytes = payload.data.tobytes()
        data_path.write_bytes(data_bytes)
        data_sha256 = compute_sha256_bytes(data_bytes)
        data_size = len(data_bytes)
        
        # Write scales if present
        scale_bytes = b""
        scale_sha256 = None
        scale_size = 0
        num_scales = 0
        
        if payload.scales is not None and payload.scales.size > 0 and scale_path:
            scale_bytes = payload.scales.tobytes()
            scale_path.write_bytes(scale_bytes)
            scale_sha256 = compute_sha256_bytes(scale_bytes)
            scale_size = len(scale_bytes)
            num_scales = payload.scales.size
        
        # Calculate bits per weight
        total_weights = np.prod(payload.original_shape)
        total_bits = (data_size + scale_size) * 8
        bits_per_weight = total_bits / total_weights if total_weights > 0 else 0.0
        
        entry = TensorIndexEntry(
            name=payload.name,
            original_shape=list(payload.original_shape),
            original_dtype=str(payload.original_dtype),
            original_size_bytes=int(np.prod(payload.original_shape) * np.dtype(payload.original_dtype).itemsize),
            encoding=payload.encoding,
            quantization=payload.quantization,
            filename=data_filename,
            offset=0,
            byte_length=data_size,
            scale_filename=scale_filename,
            scale_offset=0,
            scale_byte_length=scale_size,
            num_scales=num_scales,
            sha256=data_sha256,
            scale_sha256=scale_sha256,
            mae=payload.mae,
            mse=payload.mse,
            max_abs_error=payload.max_abs_error,
            relative_error=payload.relative_error,
            bits_per_weight=bits_per_weight,
        )
        
        return entry
    
    def _compute_storage_accounting(self) -> StorageAccounting:
        """Compute storage accounting for the artifact."""
        raw_tensor_bytes = sum(e.byte_length for e in self._tensor_entries)
        scale_bytes = sum(e.scale_byte_length for e in self._tensor_entries)
        
        # We'll compute metadata/manifest sizes after writing
        # For now, estimate parameter count
        parameter_count = self._model_info.parameter_count if self._model_info else 0
        
        # Theoretical bits per weight from compression config
        theoretical_bpw = 0.0
        if self._compression_info:
            if self._compression_info.method == CompressionMethod.INT8:
                theoretical_bpw = 8.0 + (16.0 / self._compression_info.group_size)
            elif self._compression_info.method == CompressionMethod.INT4:
                theoretical_bpw = 4.0 + (16.0 / self._compression_info.group_size)
            elif self._compression_info.method == CompressionMethod.BINARY:
                theoretical_bpw = 1.0 + (16.0 / self._compression_info.group_size)
            elif self._compression_info.method == CompressionMethod.TERNARY:
                theoretical_bpw = 1.585 + (16.0 / self._compression_info.group_size)
            elif self._compression_info.method == CompressionMethod.FP16:
                theoretical_bpw = 16.0
        
        return StorageAccounting(
            raw_tensor_bytes=raw_tensor_bytes,
            scale_bytes=scale_bytes,
            metadata_bytes=0,  # Will be updated after writing
            manifest_bytes=0,  # Will be updated after writing
            total_artifact_bytes=0,  # Will be updated after writing
            theoretical_bits_per_weight=theoretical_bpw,
            actual_bits_per_weight=0.0,  # Will be updated after writing
            parameter_count=parameter_count,
        )
    
    def _update_storage_accounting(self, manifest_bytes: int, metadata_bytes: int) -> None:
        """Update storage accounting with actual written sizes."""
        if self._storage_accounting:
            self._storage_accounting.metadata_bytes = metadata_bytes
            self._storage_accounting.manifest_bytes = manifest_bytes
            self._storage_accounting.total_artifact_bytes = (
                self._storage_accounting.raw_tensor_bytes +
                self._storage_accounting.scale_bytes +
                metadata_bytes +
                manifest_bytes
            )
            if self._storage_accounting.parameter_count > 0:
                self._storage_accounting.actual_bits_per_weight = (
                    self._storage_accounting.total_artifact_bytes * 8
                ) / self._storage_accounting.parameter_count
    
    def write(self) -> ArtifactFormat:
        """
        Write the artifact to disk.
        
        Returns:
            ArtifactFormat object representing the written artifact
        """
        # Prepare directories
        if self.artifact_dir.exists():
            if self.overwrite:
                shutil.rmtree(self.artifact_dir)
            else:
                raise FileExistsError(f"Artifact directory already exists: {self.artifact_dir}")
        
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.tensors_dir.mkdir(parents=True, exist_ok=True)
        
        # Write tensor payloads
        self._tensor_entries = []
        for i, payload in enumerate(self._tensor_payloads):
            entry = self._write_tensor_payload(payload, i)
            self._tensor_entries.append(entry)
        
        # Compute storage accounting
        self._storage_accounting = self._compute_storage_accounting()
        
        # Create manifest
        manifest = Manifest(
            format_version=self.format_version,
            model=self._model_info,
            compression=self._compression_info,
            compilation=self._compilation_info,
            hardware=self._hardware_info,
            tensors=self._tensor_entries,
            validation=self._validation_info,
            storage=self._storage_accounting,
        )
        
        # Compute manifest checksum
        manifest.manifest_sha256 = manifest.compute_sha256()
        
        # Write manifest
        manifest_json = manifest.to_json()
        manifest_bytes = len(manifest_json.encode('utf-8'))
        self.manifest_path.write_text(manifest_json)
        
        # Update storage accounting with actual manifest size
        self._update_storage_accounting(manifest_bytes, 0)
        
        # Update manifest with final storage accounting
        manifest.storage = self._storage_accounting
        manifest.manifest_sha256 = manifest.compute_sha256()
        
        # Rewrite manifest with updated storage info
        manifest_json = manifest.to_json()
        manifest_bytes = len(manifest_json.encode('utf-8'))
        self.manifest_path.write_text(manifest_json)
        
        # Final update
        self._update_storage_accounting(manifest_bytes, 0)
        
        return ArtifactFormat(manifest=manifest, artifact_path=str(self.artifact_dir))
    
    @classmethod
    def from_compilation_result(
        cls,
        artifact_dir: Union[str, Path],
        compilation_result: Any,  # CompilationResult from compiler
        format_version: ArtifactFormatVersion = ArtifactFormatVersion.V1,
        overwrite: bool = False,
    ) -> "LDMARKArtifactWriter":
        """
        Create writer from a compilation result.
        
        This is a convenience method to bridge from the compiler's
        CompilationResult to the artifact format.
        """
        writer = cls(artifact_dir, format_version, overwrite)
        
        # Extract artifact from compilation result
        if not hasattr(compilation_result, 'artifact') or not compilation_result.artifact:
            raise ValueError("Compilation result has no artifact")
        
        artifact = compilation_result.artifact
        
        # Model info
        model_info = ModelInfo(
            model_id=artifact.model_id,
            architecture=artifact.architecture or "unknown",
            parameter_count=artifact.parameter_count,
            tensor_count=len(artifact.tensors),
            original_dtype="float16",  # Default
            source_path=artifact.source_path,
        )
        writer.set_model_info(model_info)
        
        # Compression info
        comp_method = CompressionMethod(artifact.compression_method)
        compression_info = CompressionInfo(
            method=comp_method,
            target_bits=artifact.target_bits,
            group_size=artifact.group_size,
            scale_dtype=artifact.scale_dtype,
            symmetric=artifact.symmetric,
        )
        writer.set_compression_info(compression_info)
        
        # Hardware info (from compilation config if available)
        hardware_info = HardwareInfo()
        writer.set_hardware_info(hardware_info)
        
        # Compilation info
        compilation_info = create_compilation_info(
            format_version=format_version,
            optimization_strategy="balanced",
            target_precision="auto",
            config_snapshot=getattr(compilation_result, 'config_snapshot', {}),
            warnings=artifact.warnings,
        )
        writer.set_compilation_info(compilation_info)
        
        # Validation info
        validation_info = ValidationInfo(
            status=artifact.validation_status,
            passed=(artifact.validation_status == "passed"),
            details=artifact.validation_details,
        )
        writer.set_validation_info(validation_info)
        
        # Add tensors - need to load from the exported tensor files
        # For now, we'll create placeholder entries
        # The actual tensor data should be loaded from the transform_result
        if hasattr(compilation_result, 'artifact') and compilation_result.artifact:
            for tinfo in artifact.tensors:
                # Create minimal tensor entry - actual data writing happens separately
                entry = TensorIndexEntry(
                    name=tinfo.name,
                    original_shape=tinfo.original_shape,
                    original_dtype=tinfo.original_dtype,
                    original_size_bytes=tinfo.original_size_bytes,
                    encoding=TensorEncoding.GROUPWISE_QUANTIZED,
                    quantization=QuantizationParams(
                        target_bits=tinfo.target_bits,
                        group_size=tinfo.group_size,
                        scale_dtype=tinfo.scale_dtype,
                        symmetric=artifact.symmetric,
                    ),
                    filename=f"{cls._sanitize_name_static(tinfo.name)}.bin",
                    byte_length=tinfo.compressed_size_bytes,
                    scale_filename=f"{cls._sanitize_name_static(tinfo.name)}.scales.bin",
                    scale_byte_length=tinfo.scale_size_bytes,
                    num_scales=tinfo.num_scales,
                    mae=tinfo.mae,
                    mse=tinfo.mse,
                    max_abs_error=tinfo.max_abs_error,
                    relative_error=tinfo.relative_error,
                    bits_per_weight=tinfo.bits_per_weight,
                )
                writer._tensor_entries.append(entry)
        
        return writer
    
    @staticmethod
    def _sanitize_name_static(name: str) -> str:
        return name.replace(".", "_").replace("/", "_").replace("\\", "_")


def create_artifact_from_pipeline(
    artifact_dir: Union[str, Path],
    model_id: str,
    source_path: str,
    architecture: str,
    parameter_count: int,
    tensor_count: int,
    compression_method: str,
    target_bits: int,
    group_size: int,
    scale_dtype: str,
    symmetric: bool,
    tensor_infos: List[Any],  # CompressedTensorInfo from compiler
    compressed_tensors: Dict[str, Any],  # Actual tensor data from transform
    validation_status: str,
    validation_details: Dict[str, Any],
    warnings: List[str],
    format_version: ArtifactFormatVersion = ArtifactFormatVersion.V1,
    overwrite: bool = False,
    hardware_info: Optional[HardwareInfo] = None,
    compilation_config: Optional[Dict[str, Any]] = None,
) -> ArtifactFormat:
    """
    Create artifact directly from pipeline results.
    
    This is the primary entry point for the compiler's ExportStage.
    """
    writer = LDMARKArtifactWriter(artifact_dir, format_version, overwrite)
    
    # Model info
    model_info = ModelInfo(
        model_id=model_id,
        architecture=architecture,
        parameter_count=parameter_count,
        tensor_count=tensor_count,
        source_path=source_path,
    )
    writer.set_model_info(model_info)
    
    # Compression info
    comp_method = CompressionMethod(compression_method)
    compression_info = CompressionInfo(
        method=comp_method,
        target_bits=target_bits,
        group_size=group_size,
        scale_dtype=scale_dtype,
        symmetric=symmetric,
    )
    writer.set_compression_info(compression_info)
    
    # Hardware info
    writer.set_hardware_info(hardware_info or HardwareInfo())
    
    # Compilation info
    compilation_info = create_compilation_info(
        format_version=format_version,
        config_snapshot=compilation_config or {},
        warnings=warnings,
    )
    writer.set_compilation_info(compilation_info)
    
    # Validation info
    validation_info = ValidationInfo(
        status=validation_status,
        passed=(validation_status == "passed"),
        details=validation_details,
    )
    writer.set_validation_info(validation_info)
    
    # Add tensors with actual data
    for tinfo in tensor_infos:
        tdata = compressed_tensors.get(tinfo.name)
        if tdata is None:
            continue
        
        data = tdata["data"]
        scales = tdata.get("scales")
        original_shape = tuple(tdata.get("original_shape", data.shape))
        
        writer.add_tensor(
            name=tinfo.name,
            data=data,
            scales=scales,
            original_shape=original_shape,
            original_dtype=np.dtype(tinfo.original_dtype),
            encoding=TensorEncoding.GROUPWISE_QUANTIZED,
            quantization=QuantizationParams(
                target_bits=tinfo.target_bits,
                group_size=tinfo.group_size,
                scale_dtype=tinfo.scale_dtype,
                symmetric=symmetric,
            ),
            mae=tinfo.mae,
            mse=tinfo.mse,
            max_abs_error=tinfo.max_abs_error,
            relative_error=tinfo.relative_error,
        )
    
    return writer.write()