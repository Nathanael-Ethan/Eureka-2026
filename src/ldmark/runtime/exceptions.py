"""
LDMARK Runtime - Exceptions

Custom exceptions for runtime operations.
"""

from __future__ import annotations


class LDMARKRuntimeError(Exception):
    """Base exception for LDMARK runtime errors."""
    pass


class UnsupportedRuntimeFormat(LDMARKRuntimeError):
    """Raised when a compression format is not supported by the runtime."""
    
    def __init__(self, format_name: str, message: str = ""):
        self.format_name = format_name
        msg = f"Unsupported runtime format: {format_name}"
        if message:
            msg += f" - {message}"
        super().__init__(msg)


class TensorNotFoundError(LDMARKRuntimeError):
    """Raised when a tensor is not found in the artifact."""
    
    def __init__(self, tensor_name: str, available_tensors: list = None):
        self.tensor_name = tensor_name
        self.available_tensors = available_tensors or []
        msg = f"Tensor not found: {tensor_name}"
        if available_tensors:
            msg += f" (available: {len(available_tensors)} tensors)"
        super().__init__(msg)


class ArtifactCorruptedError(LDMARKRuntimeError):
    """Raised when the artifact is corrupted or malformed."""
    
    def __init__(self, message: str, tensor_name: str = ""):
        self.tensor_name = tensor_name
        super().__init__(message)


class DequantizationError(LDMARKRuntimeError):
    """Raised when tensor dequantization fails."""
    
    def __init__(self, tensor_name: str, reason: str):
        self.tensor_name = tensor_name
        super().__init__(f"Dequantization failed for {tensor_name}: {reason}")


class MemoryAccountingError(LDMARKRuntimeError):
    """Raised when memory accounting fails."""
    pass


class ComputationError(LDMARKRuntimeError):
    """Raised when a computation operation fails."""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        super().__init__(f"Computation failed ({operation}): {reason}")