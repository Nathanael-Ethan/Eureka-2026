"""
LDMARK Artifact Format

Versioned, deterministic container format for compiled models.
"""

from __future__ import annotations

from .format import (
    ArtifactFormat,
    ArtifactFormatVersion,
    Manifest,
    TensorIndexEntry,
    CompressionInfo,
    ModelInfo,
    HardwareInfo,
    ValidationInfo,
    StorageAccounting,
    CompilationInfo,
    TensorEncoding,
    QuantizationParams,
    CompressionMethod,
    create_compilation_info,
    get_ldmark_version,
)
from .writer import LDMARKArtifactWriter, create_artifact_from_pipeline
from .reader import LDMARKArtifactReader, open_artifact, inspect_artifact
from .integrity import (
    verify_artifact,
    IntegrityError,
    ChecksumMismatchError,
    SizeMismatchError,
    MissingFileError,
    MalformedManifestError,
    VersionMismatchError,
    quick_verify,
)

__all__ = [
    "ArtifactFormat",
    "ArtifactFormatVersion",
    "Manifest",
    "TensorIndexEntry",
    "CompressionInfo",
    "ModelInfo",
    "HardwareInfo",
    "ValidationInfo",
    "StorageAccounting",
    "CompilationInfo",
    "TensorEncoding",
    "QuantizationParams",
    "CompressionMethod",
    "create_compilation_info",
    "get_ldmark_version",
    "LDMARKArtifactWriter",
    "create_artifact_from_pipeline",
    "LDMARKArtifactReader",
    "open_artifact",
    "inspect_artifact",
    "verify_artifact",
    "IntegrityError",
    "ChecksumMismatchError",
    "SizeMismatchError",
    "MissingFileError",
    "MalformedManifestError",
    "VersionMismatchError",
    "quick_verify",
]