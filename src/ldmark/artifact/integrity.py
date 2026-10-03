"""
LDMARK Artifact Integrity Verification

Provides checksums, size verification, and corruption detection.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class IntegrityError(Exception):
    """Base class for integrity verification errors."""
    pass


class ChecksumMismatchError(IntegrityError):
    """Raised when a checksum verification fails."""
    def __init__(self, path: str, expected: str, actual: str):
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


class SizeMismatchError(IntegrityError):
    """Raised when file size doesn't match expected."""
    def __init__(self, path: str, expected: int, actual: int):
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(f"Size mismatch for {path}: expected {expected} bytes, got {actual}")


class MissingFileError(IntegrityError):
    """Raised when a required file is missing."""
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Required file not found: {path}")


class MalformedManifestError(IntegrityError):
    """Raised when manifest is malformed or invalid."""
    pass


class VersionMismatchError(IntegrityError):
    """Raised when format version is unsupported."""
    def __init__(self, expected: str, actual: str):
        self.expected = expected
        self.actual = actual
        super().__init__(f"Unsupported format version: expected {expected}, got {actual}")


@dataclass
class IntegrityReport:
    """Report of integrity verification results."""
    manifest_valid: bool = False
    manifest_checksum_valid: bool = False
    tensor_files_exist: bool = False
    tensor_checksums_valid: bool = False
    tensor_sizes_valid: bool = False
    scale_files_exist: bool = False
    scale_checksums_valid: bool = False
    scale_sizes_valid: bool = False
    total_files_checked: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def all_valid(self) -> bool:
        return (
            self.manifest_valid and
            self.manifest_checksum_valid and
            self.tensor_files_exist and
            self.tensor_checksums_valid and
            self.tensor_sizes_valid and
            self.scale_files_exist and
            self.scale_checksums_valid and
            self.scale_sizes_valid
        )
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        self.warnings.append(warning)


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def verify_file_size(file_path: Path, expected_size: int) -> bool:
    """Verify file size matches expected."""
    actual_size = file_path.stat().st_size
    return actual_size == expected_size


def verify_file_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verify file SHA-256 matches expected."""
    if not expected_sha256:
        return True  # No checksum to verify
    actual_sha256 = compute_sha256(file_path)
    return actual_sha256 == expected_sha256


def verify_manifest(manifest_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify manifest file integrity.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import json
        with open(manifest_path, "r") as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ["magic", "format_version", "model", "compression", "compilation", "tensors"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Check magic
        if data.get("magic") != "LDMARK":
            return False, f"Invalid magic: {data.get('magic')}"
        
        # Check format version
        from .format import ArtifactFormatVersion
        try:
            ArtifactFormatVersion.from_string(data["format_version"])
        except ValueError as e:
            return False, str(e)
        
        return True, None
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Manifest verification failed: {e}"


def verify_artifact(artifact_path: Path, verify_checksums: bool = True) -> IntegrityReport:
    """
    Verify artifact integrity.
    
    Args:
        artifact_path: Path to artifact directory
        verify_checksums: Whether to verify SHA-256 checksums
    
    Returns:
        IntegrityReport with verification results
    """
    from .format import Manifest
    
    report = IntegrityReport()
    
    # Check artifact directory exists
    if not artifact_path.exists() or not artifact_path.is_dir():
        report.add_error(f"Artifact path does not exist or is not a directory: {artifact_path}")
        return report
    
    # Find and verify manifest
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        report.add_error(f"Manifest not found: {manifest_path}")
        return report
    
    # Verify manifest structure
    valid, error = verify_manifest(manifest_path)
    report.manifest_valid = valid
    if not valid:
        report.add_error(f"Manifest invalid: {error}")
        return report
    
    # Load manifest
    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        manifest = Manifest.from_dict(manifest_data)
    except Exception as e:
        report.add_error(f"Failed to load manifest: {e}")
        return report
    
    # Verify manifest checksum
    if verify_checksums:
        computed_sha256 = manifest.compute_sha256()
        report.manifest_checksum_valid = (computed_sha256 == manifest.manifest_sha256)
        if not report.manifest_checksum_valid:
            report.add_error(f"Manifest checksum mismatch: expected {manifest.manifest_sha256}, got {computed_sha256}")
    else:
        report.manifest_checksum_valid = True
    
    # Verify tensor files
    tensor_files_exist = True
    tensor_checksums_valid = True
    tensor_sizes_valid = True
    
    for tensor in manifest.tensors:
        report.total_files_checked += 1
        tensor_path = artifact_path / "tensors" / tensor.filename
        
        if not tensor_path.exists():
            report.add_error(f"Tensor file missing: {tensor_path}")
            tensor_files_exist = False
            continue
        
        # Verify size
        if tensor.byte_length > 0:
            if not verify_file_size(tensor_path, tensor.byte_length):
                report.add_error(f"Tensor size mismatch: {tensor.filename}")
                tensor_sizes_valid = False
        
        # Verify checksum
        if verify_checksums and tensor.sha256:
            if not verify_file_sha256(tensor_path, tensor.sha256):
                report.add_error(f"Tensor checksum mismatch: {tensor.filename}")
                tensor_checksums_valid = False
    
    report.tensor_files_exist = tensor_files_exist
    report.tensor_checksums_valid = tensor_checksums_valid
    report.tensor_sizes_valid = tensor_sizes_valid
    
    # Verify scale files (if any)
    scale_files_exist = True
    scale_checksums_valid = True
    scale_sizes_valid = True
    
    for tensor in manifest.tensors:
        if tensor.scale_filename and tensor.scale_byte_length > 0:
            report.total_files_checked += 1
            scale_path = artifact_path / "tensors" / tensor.scale_filename
            
            if not scale_path.exists():
                report.add_error(f"Scale file missing: {scale_path}")
                scale_files_exist = False
                continue
            
            # Verify size
            if not verify_file_size(scale_path, tensor.scale_byte_length):
                report.add_error(f"Scale size mismatch: {tensor.scale_filename}")
                scale_sizes_valid = False
            
            # Verify checksum
            if verify_checksums and tensor.scale_sha256:
                if not verify_file_sha256(scale_path, tensor.scale_sha256):
                    report.add_error(f"Scale checksum mismatch: {tensor.scale_filename}")
                    scale_checksums_valid = False
    
    report.scale_files_exist = scale_files_exist
    report.scale_checksums_valid = scale_checksums_valid
    report.scale_sizes_valid = scale_sizes_valid
    
    return report


def quick_verify(artifact_path: Path) -> bool:
    """
    Quick verification - just check manifest exists and is valid JSON with correct magic.
    
    Does not verify tensor checksums or sizes.
    """
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        return False
    
    valid, _ = verify_manifest(manifest_path)
    return valid