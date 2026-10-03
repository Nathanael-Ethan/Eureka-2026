"""
Tensor classification utilities for mixed-precision strategies.

Classifies tensors based on name, shape, and architecture patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Pattern
import re
import numpy as np
from .plan import TensorClassification, TensorRole


@dataclass(frozen=True)
class TensorMetadata:
    """Metadata about a tensor for classification."""
    name: str
    shape: tuple
    dtype: np.dtype
    num_parameters: int
    layer_index: Optional[int] = None
    architecture: str = "unknown"
    is_weight: bool = True
    additional: Dict[str, Any] = field(default_factory=dict)


# Common naming patterns for transformer architectures
ROLE_PATTERNS: Dict[TensorRole, List[Pattern]] = {
    TensorRole.EMBEDDING: [
        re.compile(r".*(embedding|embed|wte|wpe).*"),
        re.compile(r".*(token|position)_emb.*"),
    ],
    TensorRole.ATTENTION_Q: [
        re.compile(r".*(q_proj|query|attn\.q|self_attn\.q_proj).*"),
    ],
    TensorRole.ATTENTION_K: [
        re.compile(r".*(k_proj|key|attn\.k|self_attn\.k_proj).*"),
    ],
    TensorRole.ATTENTION_V: [
        re.compile(r".*(v_proj|value|attn\.v|self_attn\.v_proj).*"),
    ],
    TensorRole.ATTENTION_O: [
        re.compile(r".*(o_proj|out_proj|attn\.out|self_attn\.o_proj).*"),
    ],
    TensorRole.ATTENTION_QKV: [
        re.compile(r".*(qkv|in_proj).*"),
    ],
    TensorRole.MLP_UP: [
        recompile(r".*(up_proj|fc1|mlp\.up|feed_forward\.up).*"),
    ],
    TensorRole.MLP_DOWN: [
        re.compile(r".*(down_proj|fc2|mlp\.down|feed_forward\.down).*"),
    ],
    TensorRole.MLP_GATE: [
        re.compile(r".*(gate_proj|mlp\.gate).*"),
    ],
    TensorRole.NORM: [
        re.compile(r".*(norm|ln_|layer_norm|layernorm|rmsnorm).*"),
        recompile(r".*(weight|bias).*norm.*"),
    ],
    TensorRole.OUTPUT_HEAD: [
        re.compile(r".*(lm_head|output|head|final_layer).*"),
    ],
    TensorRole.BIAS: [
        re.compile(r".*\.bias$"),
    ],
}


def recompile(pattern: str) -> Pattern:
    """Helper to compile regex with case-insensitive flag."""
    return re.compile(pattern, re.IGNORECASE)


def classify_tensor(
    name: str,
    shape: tuple,
    dtype: np.dtype = np.float32,
    layer_index: Optional[int] = None,
    architecture: str = "unknown",
    is_weight: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> TensorClassification:
    """
    Classify a tensor based on its name and properties.
    
    Args:
        name: Tensor name (e.g., "transformer.h.0.attn.c_attn.weight")
        shape: Tensor shape
        dtype: Data type
        layer_index: Layer index if known
        architecture: Architecture type (e.g., "gpt2", "llama", "bert")
        is_weight: Whether this is a weight (vs bias)
        metadata: Additional metadata
        
    Returns:
        TensorClassification with role and properties
    """
    num_parameters = np.prod(shape)
    
    # Determine role from name
    role = TensorRole.UNKNOWN
    name_lower = name.lower()
    
    for r, patterns in ROLE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(name_lower):
                role = r
                break
        if role != TensorRole.UNKNOWN:
            break
    
    # If still unknown but is bias, mark as bias
    if role == TensorRole.UNKNOWN and not is_weight:
        role = TensorRole.BIAS
    
    return TensorClassification(
        tensor_name=name,
        role=role,
        shape=shape,
        num_parameters=num_parameters,
        dtype=str(dtype),
        layer_index=layer_index,
        architecture=architecture,
        is_weight=is_weight,
        metadata=metadata or {},
    )


class TensorClassifier:
    """Classifies multiple tensors and provides bulk analysis."""
    
    def __init__(
        self,
        architecture: str = "unknown",
        custom_patterns: Optional[Dict[TensorRole, List[Pattern]]] = None,
    ):
        self.architecture = architecture
        self.custom_patterns = custom_patterns or {}
    
    def classify(self, metadata: TensorMetadata) -> TensorClassification:
        """Classify a single tensor."""
        return classify_tensor(
            name=metadata.name,
            shape=metadata.shape,
            dtype=metadata.dtype,
            layer_index=metadata.layer_index,
            architecture=self.architecture,
            is_weight=metadata.is_weight,
            metadata=metadata.additional,
        )
    
    def classify_model(self, tensors: Dict[str, np.ndarray]) -> Dict[str, TensorClassification]:
        """
        Classify all tensors in a model.
        
        Args:
            tensors: Dictionary of tensor name -> tensor array
            
        Returns:
            Dictionary of tensor name -> TensorClassification
        """
        results = {}
        for name, tensor in tensors.items():
            # Extract layer index from name if possible
            layer_index = self._extract_layer_index(name)
            
            meta = TensorMetadata(
                name=name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                num_parameters=tensor.size,
                layer_index=layer_index,
                architecture=self.architecture,
                is_weight=not name.endswith(".bias"),
            )
            results[name] = self.classify(meta)
        return results
    
    def _extract_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from tensor name."""
        # Common patterns: "h.0.", "layer.0.", "layers.0.", "block.0."
        patterns = [
            r"\.h\.(\d+)\.",
            r"\.layer\.(\d+)\.",
            r"\.layers\.(\d+)\.",
            r"\.block\.(\d+)\.",
            r"\.transformer\.h\.(\d+)\.",
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return int(match.group(1))
        return None
    
    def get_role_distribution(self, classifications: Dict[str, TensorClassification]) -> Dict[TensorRole, int]:
        """Get distribution of tensor roles."""
        dist = {}
        for cls in classifications.values():
            dist[cls.role] = dist.get(cls.role, 0) + 1
        return dist
    
    def get_parameter_counts_by_role(
        self, 
        classifications: Dict[str, TensorClassification]
    ) -> Dict[TensorRole, int]:
        """Get parameter counts by role."""
        counts = {}
        for cls in classifications.values():
            counts[cls.role] = counts.get(cls.role, 0) + cls.num_parameters
        return counts


def classify_model_tensors(
    tensors: Dict[str, np.ndarray],
    architecture: str = "unknown",
) -> Dict[str, TensorClassification]:
    """Convenience function to classify all tensors in a model."""
    classifier = TensorClassifier(architecture=architecture)
    return classifier.classify_model(tensors)


def estimate_layer_count(classifications: Dict[str, TensorClassification]) -> int:
    """Estimate number of layers from tensor classifications."""
    layer_indices = set()
    for cls in classifications.values():
        if cls.layer_index is not None:
            layer_indices.add(cls.layer_index)
    return max(layer_indices) + 1 if layer_indices else 0