"""
LDMARK Artifact Reader

Reads LDMARK artifact directories with support for:
- Metadata-only inspection
- Listing tensors
- Reading individual tensor payloads
- Reading compilation metadata
- Verifying integrity
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .format import (
    ArtifactFormat,
    ArtifactFormatVersion,
    Manifest,
    TensorIndexEntry,
    ModelInfo,
    CompressionInfo,
    HardwareInfo,
    CompilationInfo,
    ValidationInfo,
    StorageAccounting,
    TensorEncoding,
    QuantizationParams,
    CompressionMethod,
)
from .integrity import (
    verify_artifact,
    IntegrityError,
    ChecksumMismatchError,
    SizeMismatchError,
    MissingFileError,
    VersionMismatchError,
    quick_verify,
)


@dataclass
class TensorData:
    """Loaded tensor data with metadata."""
    name: str
    data: np.ndarray
    scales: Optional[np.ndarray]
    index_entry: TensorIndexEntry
    
    @property
    def original_shape(self) -> Tuple[int, ...]:
        return tuple(self.index_entry.original_shape)
    
    @property
    def original_dtype(self) -> np.dtype:
        return np.dtype(self.index_entry.original_dtype)
    
    @property
    def compressed_size_bytes(self) -> int:
        return self.index_entry.byte_length
    
    @property
    def scale_size_bytes(self) -> int:
        return self.index_entry.scale_byte_length


class LDMARKArtifactReader:
    """
    Reader for LDMARK compiled model artifacts.
    
    Supports lazy loading - tensors are only loaded when requested.
    """
    
    def __init__(
        self,
        artifact_dir: Union[str, Path],
        verify_checksums: bool = True,
        mmap_tensors: bool = False,
    ):
        """
        Initialize reader.
        
        Args:
            artifact_dir: Path to artifact directory
            verify_checksums: Whether to verify SHA-256 checksums on read
            mmap_tensors: Whether to memory-map tensor files (for large models)
        """
        self.artifact_dir = Path(artifact_dir).resolve()
        self.tensors_dir = self.artifact_dir / "tensors"
        self.manifest_path = self.artifact_dir / "manifest.json"
        self.verify_checksums = verify_checksums
        self.mmap_tensors = mmap_tensors
        
        self._manifest: Optional[Manifest] = None
        self._manifest_loaded = False
    
    def _load_manifest(self) -> Manifest:
        """Load and cache manifest."""
        if self._manifest_loaded and self._manifest:
            return self._manifest
        
        if not self.manifest_path.exists():
            raise MissingFileError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, "r") as f:
            data = json.load(f)
        
        # Verify format version
        try:
            format_version = ArtifactFormatVersion.from_string(data["format_version"])
        except ValueError as e:
            raise VersionMismatchError(ArtifactFormatVersion.latest().value, data["format_version"]) from e
        
        self._manifest = Manifest.from_dict(data)
        self._manifest_loaded = True
        
        # Verify manifest checksum if enabled
        if self.verify_checksums and not self._manifest.verify_integrity():
            raise ChecksumMismatchError(
                str(self.manifest_path),
                self._manifest.manifest_sha256,
                self._manifest.compute_sha256()
            )
        
        return self._manifest
    
    @property
    def manifest(self) -> Manifest:
        """Get the artifact manifest (loads if not already loaded)."""
        return self._load_manifest()
    
    def get_format_version(self) -> ArtifactFormatVersion:
        """Get artifact format version."""
        return self.manifest.format_version
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get model information."""
        return self.manifest.model
    
    def get_compression_info(self) -> Optional[CompressionInfo]:
        """Get compression configuration."""
        return self.manifest.compression
    
    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """Get target hardware information."""
        return self.manifest.hardware
    
    def get_compilation_info(self) -> Optional[CompilationInfo]:
        """Get compilation metadata."""
        return self.manifest.compilation
    
    def get_validation_info(self) -> Optional[ValidationInfo]:
        """Get validation results."""
        return self.manifest.validation
    
    def get_storage_accounting(self) -> Optional[StorageAccounting]:
        """Get storage accounting."""
        return self.manifest.storage
    
    def list_tensors(self) -> List[str]:
        """Get list of all tensor names."""
        return [t.name for t in self.manifest.tensors]
    
    def get_tensor_count(self) -> int:
        """Get number of tensors."""
        return len(self.manifest.tensors)
    
    def get_tensor_info(self, name: str) -> Optional[TensorIndexEntry]:
        """Get tensor index entry by name."""
        for t in self.manifest.tensors:
            if t.name == name:
                return t
        return None
    
    def get_tensor_index(self) -> Dict[str, TensorIndexEntry]:
        """Get full tensor index as dictionary."""
        return {t.name: t for t in self.manifest.tensors}
    
    def read_tensor(self, name: str) -> TensorData:
        """
        Read a single tensor by name.
        
        Args:
            name: Tensor name
            
        Returns:
            TensorData with loaded data and metadata
            
        Raises:
            MissingFileError: If tensor file not found
            ChecksumMismatchError: If checksum verification fails
            SizeMismatchError: If file size doesn't match
        """
        entry = self.get_tensor_info(name)
        if not entry:
            raise KeyError(f"Tensor not found in manifest: {name}")
        
        # Load tensor data
        data_path = self.tensors_dir / entry.filename
        if not data_path.exists():
            raise MissingFileError(f"Tensor file not found: {data_path}")
        
        # Verify size
        if entry.byte_length > 0:
            actual_size = data_path.stat().st_size
            if actual_size != entry.byte_length:
                raise SizeMismatchError(str(data_path), entry.byte_length, actual_size)
        
        # Verify checksum
        if self.verify_checksums and entry.sha256:
            from .integrity import verify_file_sha256
            if not verify_file_sha256(data_path, entry.sha256):
                from .integrity import compute_sha256
                actual = compute_sha256(data_path)
                raise ChecksumMismatchError(str(data_path), entry.sha256, actual)
        
        # Load data
        if self.mmap_tensors and entry.byte_length > 1024 * 1024:  # > 1MB
            data = self._mmap_load(data_path, entry)
        else:
            data = self._load_tensor_data(data_path, entry)
        
        # Load scales if present
        scales = None
        if entry.scale_filename and entry.scale_byte_length > 0:
            scale_path = self.tensors_dir / entry.scale_filename
            if not scale_path.exists():
                raise MissingFileError(f"Scale file not found: {scale_path}")
            
            # Verify size
            actual_scale_size = scale_path.stat().st_size
            if actual_scale_size != entry.scale_byte_length:
                raise SizeMismatchError(str(scale_path), entry.scale_byte_length, actual_scale_size)
            
            # Verify checksum
            if self.verify_checksums and entry.scale_sha256:
                from .integrity import verify_file_sha256
                if not verify_file_sha256(scale_path, entry.scale_sha256):
                    from .integrity import compute_sha256
                    actual = compute_sha256(scale_path)
                    raise ChecksumMismatchError(str(scale_path), entry.scale_sha256, actual)
            
            scales = self._load_tensor_data(scale_path, entry, is_scales=True)
        
        return TensorData(
            name=name,
            data=data,
            scales=scales,
            index_entry=entry,
        )
    
    def _load_tensor_data(
        self,
        path: Path,
        entry: TensorIndexEntry,
        is_scales: bool = False
    ) -> np.ndarray:
        """Load tensor data from file."""
        dtype = np.dtype(entry.quantization.scale_dtype) if is_scales and entry.quantization else np.uint8
        
        if is_scales:
            # Scales are typically float16
            return np.fromfile(path, dtype=np.float16)
        else:
            # Compressed data - raw bytes
            # The caller needs to know how to interpret based on encoding
            return np.fromfile(path, dtype=np.uint8)
    
    def _mmap_load(self, path: Path, entry: TensorIndexEntry) -> np.ndarray:
        """Memory-map tensor file for large tensors."""
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return np.frombuffer(mm, dtype=np.uint8)
    
    def read_all_tensors(self) -> Dict[str, TensorData]:
        """Read all tensors in the artifact."""
        return {name: self.read_tensor(name) for name in self.list_tensors()}
    
    def iter_tensors(self) -> Iterator[Tuple[str, TensorData]]:
        """Iterate over all tensors lazily."""
        for name in self.list_tensors():
            yield name, self.read_tensor(name)
    
    def verify_integrity(self) -> Any:
        """
        Verify artifact integrity.
        
        Returns:
            IntegrityReport with detailed results
        """
        from .integrity import verify_artifact
        return verify_artifact(self.artifact_dir, verify_checksums=self.verify_checksums)
    
    def quick_verify(self) -> bool:
        """Quick verification - just check manifest."""
        return quick_verify(self.artifact_dir)
    
    def get_artifact_format(self) -> ArtifactFormat:
        """Get complete ArtifactFormat object."""
        return ArtifactFormat(manifest=self.manifest, artifact_path=str(self.artifact_dir))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary (loads manifest only)."""
        return self.manifest.to_dict()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert artifact to JSON string."""
        return self.manifest.to_json(indent)
    
    def __enter__(self) -> "LDMARKArtifactReader":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # No cleanup needed for current implementation
        pass


def open_artifact(
    artifact_dir: Union[str, Path],
    verify_checksums: bool = True,
    mmap_tensors: bool = False,
) -> LDMARKArtifactReader:
    """
    Open an LDMARK artifact for reading.
    
    Args:
        artifact_dir: Path to artifact directory
        verify_checksums: Whether to verify SHA-256 checksums
        mmap_tensors: Whether to memory-map tensor files
        
    Returns:
        LDMARKArtifactReader instance
    """
    return LDMARKArtifactReader(artifact_dir, verify_checksums, mmap_tensors)


def inspect_artifact(artifact_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspect artifact without loading tensor data.
    
    Returns dictionary with:
    - format_version
    - model_info
    - compression_info
    - tensor_count
    - tensor_names
    - storage_accounting
    - validation_info
    """
    reader = LDMARKArtifactReader(artifact_dir, verify_checksums=False)
    manifest = reader.manifest
    
    return {
        "format_version": manifest.format_version.value,
        "model": manifest.model.to_dict() if manifest.model else None,
        "compression": manifest.compression.to_dict() if manifest.compression else None,
        "hardware": manifest.hardware.to_dict() if manifest.hardware else None,
        "compilation": manifest.compilation.to_dict() if manifest.compilation else None,
        "tensor_count": len(manifest.tensors),
        "tensor_names": [t.name for t in manifest.tensors],
        "storage": manifest.storage.to_dict() if manifest.storage else None,
        "validation": manifest.validation.to_dict() if manifest.validation else None,
        "warnings": manifest.compilation.warnings if manifest.compilation else [],
    }