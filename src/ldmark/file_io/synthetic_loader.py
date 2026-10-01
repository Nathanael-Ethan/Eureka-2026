from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .formats import ModelFormat, ModelSource
from .loaded_model import LoadedModel, LoadedTensor
from .metadata import ModelMetadata, TensorMetadata, TensorDtype
from .loader import BaseModelLoader, CorruptFileError, MalformedStateDictError, MissingDependencyError
from .capabilities import is_numpy_available


if is_numpy_available():
    import numpy as np


class SyntheticLoader(BaseModelLoader):
    def __init__(self, metadata_only: bool = False):
        super().__init__(metadata_only)
        if not is_numpy_available():
            raise MissingDependencyError("numpy is required for SyntheticLoader")

    def can_load(self, source: ModelSource) -> bool:
        return source.format == ModelFormat.SYNTHETIC

    def load(self, source: ModelSource) -> LoadedModel:
        self._validate_source(source)

        if source.is_directory:
            return self._load_from_directory(source)
        return self._load_from_file(source)

    def load_metadata(self, source: ModelSource) -> ModelMetadata:
        self._validate_source(source)

        if source.is_directory:
            return self._load_metadata_from_directory(source)
        return self._load_metadata_from_file(source)

    def _load_from_file(self, source: ModelSource) -> LoadedModel:
        try:
            with open(source.path, "r") as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptFileError(f"Invalid JSON in {source.path}: {e}")

        return self._build_from_spec(source, spec)

    def _load_from_directory(self, source: ModelSource) -> LoadedModel:
        spec_path = source.path / "synthetic_spec.json"
        if not spec_path.exists():
            raise MalformedStateDictError(f"No synthetic_spec.json found in {source.path}")

        try:
            with open(spec_path, "r") as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptFileError(f"Invalid JSON in {spec_path}: {e}")

        return self._build_from_spec(source, spec)

    def _load_metadata_from_file(self, source: ModelSource) -> ModelMetadata:
        try:
            with open(source.path, "r") as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptFileError(f"Invalid JSON in {source.path}: {e}")

        return self._build_metadata_from_spec(source, spec)

    def _load_metadata_from_directory(self, source: ModelSource) -> ModelMetadata:
        spec_path = source.path / "synthetic_spec.json"
        if not spec_path.exists():
            raise MalformedStateDictError(f"No synthetic_spec.json found in {source.path}")

        try:
            with open(spec_path, "r") as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptFileError(f"Invalid JSON in {spec_path}: {e}")

        return self._build_metadata_from_spec(source, spec)

    def _build_from_spec(self, source: ModelSource, spec: Dict[str, Any]) -> LoadedModel:
        config = spec.get("config", {})
        tensors_spec = spec.get("tensors", [])

        arch_info = self._extract_architecture_from_config(config)
        attention_type = self._detect_attention_type(config)

        tensor_metadatas = []
        loaded_tensors = {}

        for tensor_spec in tensors_spec:
            name = tensor_spec["name"]
            shape = tensor_spec["shape"]
            dtype_str = tensor_spec.get("dtype", "float16")
            dtype = self._string_to_tensor_dtype(dtype_str)

            num_elements = 1
            for dim in shape:
                num_elements *= dim

            itemsize = self._tensor_dtype_to_itemsize(dtype)
            raw_bytes = num_elements * itemsize

            tm = TensorMetadata(
                name=name,
                shape=shape,
                dtype=dtype,
                num_elements=num_elements,
                raw_bytes=raw_bytes,
            )
            tensor_metadatas.append(tm)

            if not self.metadata_only:
                arr = np.zeros(shape, dtype=self._tensor_dtype_to_numpy(dtype))
                loaded_tensors[name] = LoadedTensor(name=name, data=arr, metadata=tm)

        total_parameters = sum(tm.num_elements for tm in tensor_metadatas)

        metadata = ModelMetadata(
            model_id=spec.get("model_id", source.name),
            source_path=str(source.path),
            source_format=ModelFormat.SYNTHETIC,
            architecture=arch_info.get("architecture", "unknown"),
            num_layers=arch_info.get("num_layers"),
            hidden_size=arch_info.get("hidden_size"),
            intermediate_size=arch_info.get("intermediate_size"),
            num_attention_heads=arch_info.get("num_attention_heads"),
            num_kv_heads=arch_info.get("num_kv_heads"),
            vocab_size=arch_info.get("vocab_size"),
            max_context_length=arch_info.get("max_context_length"),
            attention_type=attention_type,
            total_parameters=total_parameters,
            tensor_count=len(tensor_metadatas),
            tensors=tensor_metadatas,
            config=config,
        )

        return LoadedModel(source=source, metadata=metadata, tensors=loaded_tensors, config=config)

    def _build_metadata_from_spec(self, source: ModelSource, spec: Dict[str, Any]) -> ModelMetadata:
        config = spec.get("config", {})
        tensors_spec = spec.get("tensors", [])

        arch_info = self._extract_architecture_from_config(config)
        attention_type = self._detect_attention_type(config)

        tensor_metadatas = []
        for tensor_spec in tensors_spec:
            name = tensor_spec["name"]
            shape = tensor_spec["shape"]
            dtype_str = tensor_spec.get("dtype", "float16")
            dtype = self._string_to_tensor_dtype(dtype_str)

            num_elements = 1
            for dim in shape:
                num_elements *= dim

            itemsize = self._tensor_dtype_to_itemsize(dtype)
            raw_bytes = num_elements * itemsize

            tm = TensorMetadata(
                name=name,
                shape=shape,
                dtype=dtype,
                num_elements=num_elements,
                raw_bytes=raw_bytes,
            )
            tensor_metadatas.append(tm)

        total_parameters = sum(tm.num_elements for tm in tensor_metadatas)

        return ModelMetadata(
            model_id=spec.get("model_id", source.name),
            source_path=str(source.path),
            source_format=ModelFormat.SYNTHETIC,
            architecture=arch_info.get("architecture", "unknown"),
            num_layers=arch_info.get("num_layers"),
            hidden_size=arch_info.get("hidden_size"),
            intermediate_size=arch_info.get("intermediate_size"),
            num_attention_heads=arch_info.get("num_attention_heads"),
            num_kv_heads=arch_info.get("num_kv_heads"),
            vocab_size=arch_info.get("vocab_size"),
            max_context_length=arch_info.get("max_context_length"),
            attention_type=attention_type,
            total_parameters=total_parameters,
            tensor_count=len(tensor_metadatas),
            tensors=tensor_metadatas,
            config=config,
        )

    def _string_to_tensor_dtype(self, dtype_str: str) -> TensorDtype:
        dtype_map = {
            "float32": TensorDtype.FLOAT32,
            "float16": TensorDtype.FLOAT16,
            "bfloat16": TensorDtype.BFLOAT16,
            "int8": TensorDtype.INT8,
            "uint8": TensorDtype.UINT8,
            "float8": TensorDtype.FLOAT8,
            "float4": TensorDtype.FLOAT4,
            "bool": TensorDtype.BOOL,
        }
        return dtype_map.get(dtype_str.lower(), TensorDtype.UNKNOWN)

    def _tensor_dtype_to_numpy(self, dtype: TensorDtype):
        dtype_map = {
            TensorDtype.FLOAT32: np.float32,
            TensorDtype.FLOAT16: np.float16,
            TensorDtype.INT8: np.int8,
            TensorDtype.UINT8: np.uint8,
            TensorDtype.BOOL: np.bool_,
        }
        return dtype_map.get(dtype, np.float16)

    def _tensor_dtype_to_itemsize(self, dtype: TensorDtype) -> int:
        itemsize_map = {
            TensorDtype.FLOAT32: 4,
            TensorDtype.FLOAT16: 2,
            TensorDtype.BFLOAT16: 2,
            TensorDtype.INT8: 1,
            TensorDtype.UINT8: 1,
            TensorDtype.FLOAT8: 1,
            TensorDtype.FLOAT4: 1,
            TensorDtype.BOOL: 1,
            TensorDtype.UNKNOWN: 0,
        }
        return itemsize_map.get(dtype, 0)