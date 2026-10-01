from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np

from .formats import ModelFormat, ModelSource
from .metadata import ModelMetadata, TensorMetadata, TensorDtype


@dataclass
class LoadedTensor:
    name: str
    data: np.ndarray
    metadata: TensorMetadata

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> TensorDtype:
        return self.metadata.dtype

    @property
    def num_elements(self) -> int:
        return self.data.size


class LoadedModel:
    def __init__(
        self,
        source: ModelSource,
        metadata: ModelMetadata,
        tensors: Optional[Dict[str, LoadedTensor]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.metadata = metadata
        self._tensors: Dict[str, LoadedTensor] = tensors or {}
        self.config = config or metadata.config

    @property
    def model_id(self) -> str:
        return self.metadata.model_id

    @property
    def source_format(self) -> ModelFormat:
        return self.source.format

    @property
    def tensor_names(self) -> List[str]:
        return list(self._tensors.keys())

    @property
    def tensor_count(self) -> int:
        return len(self._tensors)

    @property
    def total_parameters(self) -> int:
        return self.metadata.total_parameters

    def get_tensor(self, name: str) -> Optional[LoadedTensor]:
        return self._tensors.get(name)

    def get_tensor_data(self, name: str) -> Optional[np.ndarray]:
        tensor = self._tensors.get(name)
        return tensor.data if tensor else None

    def get_tensor_metadata(self, name: str) -> Optional[TensorMetadata]:
        tensor = self._tensors.get(name)
        return tensor.metadata if tensor else None

    def iter_tensors(self) -> Iterator[Tuple[str, LoadedTensor]]:
        for name, tensor in self._tensors.items():
            yield name, tensor

    def to_state_dict(self) -> Dict[str, np.ndarray]:
        return {name: tensor.data for name, tensor in self._tensors.items()}

    def to_metadata_dict(self) -> Dict[str, Any]:
        return self.metadata.to_dict()

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_format": self.source_format.value,
            "source_path": str(self.source.path),
            "architecture": self.metadata.architecture,
            "total_parameters": self.metadata.total_parameters,
            "tensor_count": self.metadata.tensor_count,
            "num_layers": self.metadata.num_layers,
            "hidden_size": self.metadata.hidden_size,
            "vocab_size": self.metadata.vocab_size,
        }

    def __repr__(self) -> str:
        return (
            f"LoadedModel("
            f"model_id={self.model_id!r}, "
            f"format={self.source_format.value}, "
            f"params={self.total_parameters:,}, "
            f"tensors={self.tensor_count}"
            f")"
        )