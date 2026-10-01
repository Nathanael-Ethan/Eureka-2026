from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .formats import ModelFormat, ModelSource
from .loaded_model import LoadedModel, LoadedTensor
from .metadata import ModelMetadata, TensorMetadata, TensorDtype
from .loader import BaseModelLoader, CorruptFileError, MissingDependencyError, MalformedStateDictError
from .capabilities import is_safetensors_available, is_numpy_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.numpy import load_file as safetensors_load_file

if is_numpy_available():
    import numpy as np


class SafetensorsLoader(BaseModelLoader):
    def __init__(self, metadata_only: bool = False):
        super().__init__(metadata_only)
        if not is_safetensors_available():
            raise MissingDependencyError(
                "safetensors is required for SafetensorsLoader. Install with: pip install safetensors"
            )

    def can_load(self, source: ModelSource) -> bool:
        return source.format == ModelFormat.SAFETENSORS

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
            with safe_open(source.path, framework="np", device="cpu") as f:
                tensor_names = list(f.keys())
                tensors_dict = {}
                tensor_metadatas = []

                for name in tensor_names:
                    tensor = f.get_tensor(name)
                    shape = list(tensor.shape)
                    dtype = self._numpy_dtype_to_tensor_dtype(tensor.dtype)
                    num_elements = tensor.size
                    raw_bytes = num_elements * tensor.dtype.itemsize

                    tm = TensorMetadata(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        num_elements=num_elements,
                        raw_bytes=raw_bytes,
                    )
                    tensor_metadatas.append(tm)

                    if not self.metadata_only:
                        tensors_dict[name] = tensor

                config = self._load_config_from_directory(source)
                return self._build_loaded_model(source, tensors_dict, tensor_metadatas, config)
        except Exception as e:
            raise CorruptFileError(f"Failed to load safetensors file {source.path}: {e}")

    def _load_from_directory(self, source: ModelSource) -> LoadedModel:
        st_files = sorted(source.path.glob("*.safetensors"))

        if not st_files:
            raise MalformedStateDictError(f"No .safetensors files found in {source.path}")

        tensors_dict = {}
        tensor_metadatas = []

        for st_file in st_files:
            try:
                with safe_open(st_file, framework="np", device="cpu") as f:
                    for name in f.keys():
                        tensor = f.get_tensor(name)
                        shape = list(tensor.shape)
                        dtype = self._numpy_dtype_to_tensor_dtype(tensor.dtype)
                        num_elements = tensor.size
                        raw_bytes = num_elements * tensor.dtype.itemsize

                        tm = TensorMetadata(
                            name=name,
                            shape=shape,
                            dtype=dtype,
                            num_elements=num_elements,
                            raw_bytes=raw_bytes,
                        )
                        tensor_metadatas.append(tm)

                        if not self.metadata_only:
                            tensors_dict[name] = tensor
            except Exception as e:
                raise CorruptFileError(f"Failed to load {st_file}: {e}")

        config = self._load_config_from_directory(source)
        return self._build_loaded_model(source, tensors_dict, tensor_metadatas, config)

    def _load_metadata_from_file(self, source: ModelSource) -> ModelMetadata:
        try:
            with safe_open(source.path, framework="np", device="cpu") as f:
                tensor_names = list(f.keys())
                tensor_metadatas = []

                for name in tensor_names:
                    tensor = f.get_slice(name)
                    shape = list(tensor.get_shape())
                    dtype_str = tensor.get_dtype()
                    dtype = self._safetensors_dtype_to_tensor_dtype(dtype_str)
                    num_elements = 1
                    for dim in shape:
                        num_elements *= dim
                    raw_bytes = num_elements * self._dtype_str_to_itemsize(dtype_str)

                    tm = TensorMetadata(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        num_elements=num_elements,
                        raw_bytes=raw_bytes,
                    )
                    tensor_metadatas.append(tm)

                config = self._load_config_from_directory(source)
                return self._build_metadata(source, tensor_metadatas, config)
        except Exception as e:
            raise CorruptFileError(f"Failed to load safetensors metadata from {source.path}: {e}")

    def _load_metadata_from_directory(self, source: ModelSource) -> ModelMetadata:
        st_files = sorted(source.path.glob("*.safetensors"))

        if not st_files:
            raise MalformedStateDictError(f"No .safetensors files found in {source.path}")

        tensor_metadatas = []

        for st_file in st_files:
            try:
                with safe_open(st_file, framework="np", device="cpu") as f:
                    for name in f.keys():
                        tensor = f.get_slice(name)
                        shape = list(tensor.get_shape())
                        dtype_str = tensor.get_dtype()
                        dtype = self._safetensors_dtype_to_tensor_dtype(dtype_str)
                        num_elements = 1
                        for dim in shape:
                            num_elements *= dim
                        raw_bytes = num_elements * self._dtype_str_to_itemsize(dtype_str)

                        tm = TensorMetadata(
                            name=name,
                            shape=shape,
                            dtype=dtype,
                            num_elements=num_elements,
                            raw_bytes=raw_bytes,
                        )
                        tensor_metadatas.append(tm)
            except Exception as e:
                raise CorruptFileError(f"Failed to load metadata from {st_file}: {e}")

        config = self._load_config_from_directory(source)
        return self._build_metadata(source, tensor_metadatas, config)

    def _build_loaded_model(
        self,
        source: ModelSource,
        tensors_dict: Dict[str, Any],
        tensor_metadatas: List[TensorMetadata],
        config: Dict[str, Any],
    ) -> LoadedModel:
        arch_info = self._extract_architecture_from_config(config) if config else {}
        total_parameters = sum(tm.num_elements for tm in tensor_metadatas)

        metadata = ModelMetadata(
            model_id=source.name,
            source_path=str(source.path),
            source_format=ModelFormat.SAFETENSORS,
            architecture=arch_info.get("architecture", "unknown"),
            num_layers=arch_info.get("num_layers"),
            hidden_size=arch_info.get("hidden_size"),
            intermediate_size=arch_info.get("intermediate_size"),
            num_attention_heads=arch_info.get("num_attention_heads"),
            num_kv_heads=arch_info.get("num_kv_heads"),
            vocab_size=arch_info.get("vocab_size"),
            max_context_length=arch_info.get("max_context_length"),
            attention_type=self._detect_attention_type(config) if config else None,
            total_parameters=total_parameters,
            tensor_count=len(tensor_metadatas),
            tensors=tensor_metadatas,
            config=config,
        )

        loaded_tensors = {}
        if not self.metadata_only:
            for name, data in tensors_dict.items():
                tm = next((t for t in tensor_metadatas if t.name == name), None)
                if tm:
                    loaded_tensors[name] = LoadedTensor(name=name, data=data, metadata=tm)

        return LoadedModel(source=source, metadata=metadata, tensors=loaded_tensors, config=config)

    def _build_metadata(
        self,
        source: ModelSource,
        tensor_metadatas: List[TensorMetadata],
        config: Dict[str, Any],
    ) -> ModelMetadata:
        arch_info = self._extract_architecture_from_config(config) if config else {}
        total_parameters = sum(tm.num_elements for tm in tensor_metadatas)

        return ModelMetadata(
            model_id=source.name,
            source_path=str(source.path),
            source_format=ModelFormat.SAFETENSORS,
            architecture=arch_info.get("architecture", "unknown"),
            num_layers=arch_info.get("num_layers"),
            hidden_size=arch_info.get("hidden_size"),
            intermediate_size=arch_info.get("intermediate_size"),
            num_attention_heads=arch_info.get("num_attention_heads"),
            num_kv_heads=arch_info.get("num_kv_heads"),
            vocab_size=arch_info.get("vocab_size"),
            max_context_length=arch_info.get("max_context_length"),
            attention_type=self._detect_attention_type(config) if config else None,
            total_parameters=total_parameters,
            tensor_count=len(tensor_metadatas),
            tensors=tensor_metadatas,
            config=config,
        )

    def _numpy_dtype_to_tensor_dtype(self, dtype) -> TensorDtype:
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

    def _safetensors_dtype_to_tensor_dtype(self, dtype_str: str) -> TensorDtype:
        dtype_map = {
            "F32": TensorDtype.FLOAT32,
            "F16": TensorDtype.FLOAT16,
            "BF16": TensorDtype.BFLOAT16,
            "I8": TensorDtype.INT8,
            "U8": TensorDtype.UINT8,
            "BOOL": TensorDtype.BOOL,
            "F8_E4M3": TensorDtype.FLOAT8,
            "F8_E5M2": TensorDtype.FLOAT8,
            "F4": TensorDtype.FLOAT4,
        }
        return dtype_map.get(dtype_str, TensorDtype.UNKNOWN)

    def _dtype_str_to_itemsize(self, dtype_str: str) -> int:
        itemsize_map = {
            "F32": 4,
            "F16": 2,
            "BF16": 2,
            "I8": 1,
            "U8": 1,
            "BOOL": 1,
            "F8_E4M3": 1,
            "F8_E5M2": 1,
            "F4": 1,
        }
        return itemsize_map.get(dtype_str, 0)

    def _load_config_from_directory(self, source: ModelSource) -> Dict[str, Any]:
        if not source.is_directory:
            return {}

        config_path = source.path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return {}