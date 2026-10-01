from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .formats import ModelFormat, ModelSource, detect_format_from_path
from .loaded_model import LoadedModel
from .metadata import ModelMetadata, TensorMetadata, TensorDtype


class ModelLoaderError(Exception):
    pass


class UnsupportedFormatError(ModelLoaderError):
    pass


class CorruptFileError(ModelLoaderError):
    pass


class MissingDependencyError(ModelLoaderError):
    pass


class MalformedStateDictError(ModelLoaderError):
    pass


class ModelLoader(ABC):
    def __init__(self, metadata_only: bool = False):
        self.metadata_only = metadata_only

    @abstractmethod
    def can_load(self, source: ModelSource) -> bool:
        pass

    @abstractmethod
    def load(self, source: ModelSource) -> LoadedModel:
        pass

    @abstractmethod
    def load_metadata(self, source: ModelSource) -> ModelMetadata:
        pass

    def load_from_path(self, path: Union[str, Path]) -> LoadedModel:
        source = detect_format_from_path(path)
        return self.load(source)

    def load_metadata_from_path(self, path: Union[str, Path]) -> ModelMetadata:
        source = detect_format_from_path(path)
        return self.load_metadata(source)


class BaseModelLoader(ModelLoader):
    def __init__(self, metadata_only: bool = False):
        super().__init__(metadata_only)

    def can_load(self, source: ModelSource) -> bool:
        return False

    def load(self, source: ModelSource) -> LoadedModel:
        raise NotImplementedError

    def load_metadata(self, source: ModelSource) -> ModelMetadata:
        raise NotImplementedError

    def _validate_source(self, source: ModelSource) -> None:
        if not source.path.exists():
            raise FileNotFoundError(f"Path does not exist: {source.path}")

    def _extract_architecture_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        arch = config.get("architectures", [])
        architecture = arch[0] if arch and arch[0] else config.get("model_type", "unknown")

        return {
            "architecture": architecture if architecture else "unknown",
            "num_layers": config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers"),
            "hidden_size": config.get("hidden_size") or config.get("n_embd") or config.get("d_model"),
            "intermediate_size": config.get("intermediate_size") or config.get("n_inner") or config.get("ffn_dim"),
            "num_attention_heads": config.get("num_attention_heads") or config.get("n_head") or config.get("num_heads"),
            "num_kv_heads": config.get("num_key_value_heads") or config.get("n_kv_head"),
            "vocab_size": config.get("vocab_size") or config.get("n_vocab"),
            "max_context_length": config.get("max_position_embeddings") or config.get("n_positions") or config.get("max_seq_len"),
        }

    def _detect_attention_type(self, config: Dict[str, Any]) -> Optional[str]:
        num_heads = config.get("num_attention_heads") or config.get("n_head")
        num_kv_heads = config.get("num_key_value_heads") or config.get("n_kv_head")

        if num_heads and num_kv_heads:
            if num_kv_heads == 1:
                return "mqa"
            elif num_kv_heads < num_heads:
                return "gqa"
            else:
                return "mha"
        return None

    def _numpy_dtype_to_tensor_dtype(self, dtype) -> TensorDtype:
        import numpy as np

        dtype_map = {
            np.float32: TensorDtype.FLOAT32,
            np.float16: TensorDtype.FLOAT16,
            np.int8: TensorDtype.INT8,
            np.uint8: TensorDtype.UINT8,
            np.bool_: TensorDtype.BOOL,
        }

        if hasattr(np, "bfloat16"):
            dtype_map[np.bfloat16] = TensorDtype.BFLOAT16

        return dtype_map.get(dtype.type, TensorDtype.UNKNOWN)

    def _infer_weight_dtype(self, tensors: Dict[str, np.ndarray]) -> TensorDtype:
        if not tensors:
            return TensorDtype.UNKNOWN

        from collections import Counter
        dtype_counts = Counter()
        for arr in tensors.values():
            dtype = self._numpy_dtype_to_tensor_dtype(arr.dtype)
            dtype_counts[dtype] += arr.size

        if dtype_counts:
            return dtype_counts.most_common(1)[0][0]
        return TensorDtype.UNKNOWN