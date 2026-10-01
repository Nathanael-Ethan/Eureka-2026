from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .formats import ModelFormat, ModelSource
from .loaded_model import LoadedModel, LoadedTensor
from .metadata import ModelMetadata, TensorMetadata, TensorDtype
from .loader import BaseModelLoader, UnsupportedFormatError, CorruptFileError, MalformedStateDictError
from .capabilities import is_numpy_available


if is_numpy_available():
    import numpy as np


class NumpyLoader(BaseModelLoader):
    def __init__(self, metadata_only: bool = False):
        super().__init__(metadata_only)
        if not is_numpy_available():
            raise MissingDependencyError("numpy is required for NumpyLoader")

    def can_load(self, source: ModelSource) -> bool:
        return source.format == ModelFormat.NUMPY

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
            data = np.load(source.path, allow_pickle=False)
        except Exception as e:
            raise CorruptFileError(f"Failed to load NumPy file {source.path}: {e}")

        if isinstance(data, np.lib.npyio.NpzFile):
            tensors_dict = {key: data[key] for key in data.files}
            data.close()
        elif isinstance(data, np.ndarray):
            tensors_dict = {source.path.stem: data}
        else:
            raise MalformedStateDictError(f"Unexpected NumPy file format in {source.path}")

        return self._build_loaded_model(source, tensors_dict)

    def _load_from_directory(self, source: ModelSource) -> LoadedModel:
        npy_files = sorted(source.path.glob("*.npy"))
        npz_files = sorted(source.path.glob("*.npz"))

        if not npy_files and not npz_files:
            raise MalformedStateDictError(f"No .npy or .npz files found in {source.path}")

        tensors_dict = {}

        for npy_file in npy_files:
            try:
                arr = np.load(npy_file, allow_pickle=False)
                tensors_dict[npy_file.stem] = arr
            except Exception as e:
                raise CorruptFileError(f"Failed to load {npy_file}: {e}")

        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=False)
                for key in data.files:
                    tensors_dict[f"{npz_file.stem}.{key}"] = data[key]
                data.close()
            except Exception as e:
                raise CorruptFileError(f"Failed to load {npz_file}: {e}")

        return self._build_loaded_model(source, tensors_dict)

    def _load_metadata_from_file(self, source: ModelSource) -> ModelMetadata:
        try:
            data = np.load(source.path, allow_pickle=False)
        except Exception as e:
            raise CorruptFileError(f"Failed to load NumPy file {source.path}: {e}")

        if isinstance(data, np.lib.npyio.NpzFile):
            tensor_names = data.files
            shapes = {key: data[key].shape for key in tensor_names}
            dtypes = {key: data[key].dtype for key in tensor_names}
            data.close()
        elif isinstance(data, np.ndarray):
            tensor_names = [source.path.stem]
            shapes = {source.path.stem: data.shape}
            dtypes = {source.path.stem: data.dtype}
        else:
            raise MalformedStateDictError(f"Unexpected NumPy file format in {source.path}")

        return self._build_metadata(source, tensor_names, shapes, dtypes, {})

    def _load_metadata_from_directory(self, source: ModelSource) -> ModelMetadata:
        npy_files = sorted(source.path.glob("*.npy"))
        npz_files = sorted(source.path.glob("*.npz"))

        if not npy_files and not npz_files:
            raise MalformedStateDictError(f"No .npy or .npz files found in {source.path}")

        tensor_names = []
        shapes = {}
        dtypes = {}

        for npy_file in npy_files:
            arr = np.load(npy_file, allow_pickle=False, mmap_mode="r")
            tensor_names.append(npy_file.stem)
            shapes[npy_file.stem] = arr.shape
            dtypes[npy_file.stem] = arr.dtype

        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=False)
            for key in data.files:
                name = f"{npz_file.stem}.{key}"
                tensor_names.append(name)
                shapes[name] = data[key].shape
                dtypes[name] = data[key].dtype
            data.close()

        return self._build_metadata(source, tensor_names, shapes, dtypes, {})

    def _build_loaded_model(self, source: ModelSource, tensors_dict: Dict[str, Any]) -> LoadedModel:
        config = self._load_config_from_directory(source)
        arch_info = self._extract_architecture_from_config(config) if config else {}

        tensor_metadatas = []
        loaded_tensors = {}

        for name, arr in tensors_dict.items():
            if not isinstance(arr, np.ndarray):
                raise MalformedStateDictError(f"Tensor {name} is not a numpy array")

            dtype = self._numpy_dtype_to_tensor_dtype(arr.dtype)
            num_elements = arr.size
            raw_bytes = num_elements * arr.dtype.itemsize

            tm = TensorMetadata(
                name=name,
                shape=list(arr.shape),
                dtype=dtype,
                num_elements=num_elements,
                raw_bytes=raw_bytes,
            )
            tensor_metadatas.append(tm)

            if not self.metadata_only:
                loaded_tensors[name] = LoadedTensor(name=name, data=arr, metadata=tm)

        total_parameters = sum(tm.num_elements for tm in tensor_metadatas)

        metadata = ModelMetadata(
            model_id=source.path.stem,
            source_path=str(source.path),
            source_format=ModelFormat.NUMPY,
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

        return LoadedModel(source=source, metadata=metadata, tensors=loaded_tensors, config=config)

    def _build_metadata(
        self,
        source: ModelSource,
        tensor_names: List[str],
        shapes: Dict[str, tuple],
        dtypes: Dict[str, Any],
        config: Dict[str, Any],
    ) -> ModelMetadata:
        arch_info = self._extract_architecture_from_config(config) if config else {}

        tensor_metadatas = []
        for name in tensor_names:
            shape = list(shapes[name])
            dtype = self._numpy_dtype_to_tensor_dtype(dtypes[name])
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            raw_bytes = num_elements * dtypes[name].itemsize

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
            model_id=source.path.stem,
            source_path=str(source.path),
            source_format=ModelFormat.NUMPY,
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