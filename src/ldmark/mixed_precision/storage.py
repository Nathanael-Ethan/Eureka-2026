"""
Storage accounting for mixed-precision strategies.

Calculates total model storage including all overheads.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from .plan import (
    MixedPrecisionPlan, 
    TensorPrecisionAssignment, 
    PrecisionType, 
    PrecisionCandidate,
    TensorClassification,
)


@dataclass(frozen=True)
class StorageComponent:
    """Storage breakdown for a single component."""
    tensor_name: str
    precision: PrecisionType
    weight_bytes: int
    scale_bytes: int
    metadata_bytes: int
    total_bytes: int
    bits_per_weight: float
    group_size: Optional[int]
    codebook_size: Optional[int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "precision": self.precision.value,
            "weight_bytes": self.weight_bytes,
            "scale_bytes": self.scale_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "bits_per_weight": self.bits_per_weight,
            "group_size": self.group_size,
            "codebook_size": self.codebook_size,
        }


@dataclass(frozen=True)
class StorageAccounting:
    """Complete storage accounting for a mixed-precision plan."""
    components: List[StorageComponent]
    total_bytes: int
    total_parameters: int
    average_bits_per_weight: float
    compression_ratio: float  # vs FP32 baseline
    precision_distribution: Dict[str, int]
    overhead_bytes: int
    overhead_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": [c.to_dict() for c in self.components],
            "total_bytes": self.total_bytes,
            "total_parameters": self.total_parameters,
            "average_bits_per_weight": self.average_bits_per_weight,
            "compression_ratio": self.compression_ratio,
            "precision_distribution": self.precision_distribution,
            "overhead_bytes": self.overhead_bytes,
            "overhead_percentage": self.overhead_percentage,
        }
    
    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)


class MixedPrecisionStorageModel:
    """
    Storage model for mixed-precision plans.
    
    Calculates actual storage including:
    - Weight storage (packed indices or direct values)
    - Scale factors (group-wise)
    - Codebooks (for codebook-based quantization)
    - Metadata (tensor headers, alignment padding)
    - Artifact format overhead
    """
    
    # Estimated per-tensor overhead in bytes
    TENSOR_HEADER_BYTES = 64  # name, shape, dtype, precision info
    ALIGNMENT_BYTES = 16  # typical alignment padding
    
    def __init__(
        self,
        group_size: int = 128,
        scale_dtype: np.dtype = np.float16,
        tensor_header_bytes: int = TENSOR_HEADER_BYTES,
        alignment_bytes: int = ALIGNMENT_BYTES,
        artifact_overhead_factor: float = 1.02,  # 2% format overhead
    ):
        self.group_size = group_size
        self.scale_dtype = scale_dtype
        self.scale_bytes = scale_dtype.itemsize
        self.tensor_header_bytes = tensor_header_bytes
        self.alignment_bytes = alignment_bytes
        self.artifact_overhead_factor = artifact_overhead_factor
    
    def calculate_tensor_storage(
        self,
        assignment: TensorPrecisionAssignment,
        tensor_classification: Optional[TensorClassification] = None,
    ) -> StorageComponent:
        """Calculate storage for a single tensor."""
        prec = assignment.precision
        num_params = assignment.candidate.estimated_storage_bytes if assignment.candidate else 0
        
        if prec == PrecisionType.FP32:
            weight_bytes = num_params * 4
            scale_bytes = 0
            bits_per_weight = 32.0
            
        elif prec == PrecisionType.FP16:
            weight_bytes = num_params * 2
            scale_bytes = 0
            bits_per_weight = 16.0
            
        elif prec == PrecisionType.INT8:
            # Group-wise INT8
            group_size = assignment.group_size or self.group_size
            num_groups = (num_params + group_size - 1) // group_size
            weight_bytes = num_params  # 1 byte per weight
            scale_bytes = num_groups * self.scale_bytes
            bits_per_weight = (weight_bytes * 8 + scale_bytes * 8) / num_params
            
        elif prec == PrecisionType.INT4:
            # Group-wise INT4 (packed)
            group_size = assignment.group_size or self.group_size
            num_groups = (num_params + group_size - 1) // group_size
            weight_bytes = (num_params + 1) // 2  # 2 weights per byte
            scale_bytes = num_groups * self.scale_bytes
            bits_per_weight = (weight_bytes * 8 + scale_bytes * 8) / num_params
            
        elif prec == PrecisionType.BINARY:
            # Binary: 1 bit per weight + scale
            group_size = assignment.group_size or self.group_size
            num_groups = (num_params + group_size - 1) // group_size
            weight_bytes = (num_params + 7) // 8
            scale_bytes = num_groups * self.scale_bytes
            bits_per_weight = (weight_bytes * 8 + scale_bytes * 8) / num_params
            
        elif prec == PrecisionType.TERNARY:
            # Ternary: 2 bits per weight (4 values in 2 bits) + scale
            group_size = assignment.group_size or self.group_size
            num_groups = (num_params + group_size - 1) // group_size
            weight_bytes = (num_params * 2 + 7) // 8
            scale_bytes = num_groups * self.scale_bytes
            bits_per_weight = (weight_bytes * 8 + scale_bytes * 8) / num_params
            
        else:
            raise ValueError(f"Unknown precision type: {prec}")
        
        # Metadata overhead
        metadata_bytes = self.tensor_header_bytes + self.alignment_bytes
        
        total_bytes = weight_bytes + scale_bytes + metadata_bytes
        
        return StorageComponent(
            tensor_name="",  # Will be set by caller
            precision=prec,
            weight_bytes=weight_bytes,
            scale_bytes=scale_bytes,
            metadata_bytes=metadata_bytes,
            total_bytes=total_bytes,
            bits_per_weight=bits_per_weight,
            group_size=assignment.group_size,
            codebook_size=assignment.codebook_size,
        )
    
    def calculate_plan_storage(
        self,
        plan: MixedPrecisionPlan,
        classifications: Dict[str, TensorClassification],
    ) -> StorageAccounting:
        """Calculate total storage for a mixed-precision plan."""
        components = []
        total_bytes = 0
        total_parameters = 0
        weight_bytes_total = 0
        overhead_bytes_total = 0
        precision_dist = {}
        
        for tensor_name, assignment in plan.assignments.items():
            classification = classifications.get(tensor_name)
            num_params = classification.num_parameters if classification else 0
            total_parameters += num_params
            
            component = self.calculate_tensor_storage(assignment, classification)
            component = StorageComponent(
                tensor_name=tensor_name,
                precision=component.precision,
                weight_bytes=component.weight_bytes,
                scale_bytes=component.scale_bytes,
                metadata_bytes=component.metadata_bytes,
                total_bytes=component.total_bytes,
                bits_per_weight=component.bits_per_weight,
                group_size=component.group_size,
                codebook_size=component.codebook_size,
            )
            components.append(component)
            
            total_bytes += component.total_bytes
            weight_bytes_total += component.weight_bytes
            overhead_bytes_total += component.scale_bytes + component.metadata_bytes
            
            prec_key = component.precision.value
            precision_dist[prec_key] = precision_dist.get(prec_key, 0) + 1
        
        # Apply artifact overhead factor
        total_bytes = int(total_bytes * self.artifact_overhead_factor)
        
        # Average bits per weight
        avg_bpw = (total_bytes * 8) / total_parameters if total_parameters > 0 else 0
        
        # Compression ratio vs FP32
        fp32_bytes = total_parameters * 4
        compression_ratio = fp32_bytes / total_bytes if total_bytes > 0 else float('inf')
        
        overhead_percentage = (overhead_bytes_total / total_bytes * 100) if total_bytes > 0 else 0
        
        return StorageAccounting(
            components=components,
            total_bytes=total_bytes,
            total_parameters=total_parameters,
            average_bits_per_weight=avg_bpw,
            compression_ratio=compression_ratio,
            precision_distribution=precision_dist,
            overhead_bytes=overhead_bytes_total,
            overhead_percentage=overhead_percentage,
        )


def calculate_mixed_precision_storage(
    plan: MixedPrecisionPlan,
    classifications: Dict[str, TensorClassification],
    group_size: int = 128,
) -> StorageAccounting:
    """Convenience function to calculate storage for a plan."""
    model = MixedPrecisionStorageModel(group_size=group_size)
    return model.calculate_plan_storage(plan, classifications)


def calculate_uniform_storage(
    classifications: Dict[str, TensorClassification],
    precision: PrecisionType,
    group_size: int = 128,
) -> StorageAccounting:
    """Calculate storage for a uniform precision plan."""
    tensor_names = list(classifications.keys())
    plan = create_uniform_plan("uniform", tensor_names, precision, group_size)
    return calculate_mixed_precision_storage(plan, classifications, group_size)