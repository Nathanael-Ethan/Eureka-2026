from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .formats import ModelFormat, ModelSource
from .loaded_model import LoadedModel, LoadedTensor
from .metadata import ModelMetadata, TensorMetadata, TensorDtype
from .loader import BaseModelLoader, UnsupportedFormatError, CorruptFileError, MissingDependencyError, MalformedStateDictError
from .capabilities import is_torch_available, is_numpy_available


if is_torch_available():
    import torch

if is_numpy_available():
    import numpy as np


class TorchLoader(BaseModelLoader):
    def __init__(self, metadata_only: bool = False):
        super().__init__(metadata_only)
        if not is_torch_available():
            raise MissingDependencyError(
                "PyTorch is required for TorchLoader. Install with: pip install torch"
            )

    def can_load(self, source: ModelSource) -> bool:
        return source.format == ModelFormat.PYTORCH

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
            data = torch.load(source.path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise CorruptFileError(f"Failed to load PyTorch file {source.path}: {e}")

        tensors_dict = self._extract_state_dict(data)
        return self._build_loaded_model(source, tensors_dict)

    def _load_from_directory(self, source: ModelSource) -> LoadedModel:
        bin_files = sorted(source.path.glob("*.bin"))
        pt_files = sorted(source.path.glob("*.pt"))
        pth_files = sorted(source.path.glob("*.pth"))

        all_files = bin_files + pt_files + pth_files

        if not all_files:
            raise MalformedStateDictError(f"No PyTorch model files found in {source.path}")

        tensors_dict = {}
        for file_path in all_files:
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                file_tensors = self._extract_state_dict(data)
                for name, tensor in file_tensors.items():
                    tensors_dict[f"{file_path.stem}.{name}"] = tensor
            except Exception as e:
                raise CorruptFileError(f"Failed to load {file_path}: {e}")

        return self._build_loaded_model(source, tensors_dict)

    def _load_metadata_from_file(self, source: ModelSource) -> ModelMetadata:
        try:
            data = torch.load(source.path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise CorruptFileError(f"Failed to load PyTorch file {source.path}: {e}")

        tensors_dict = self._extract_state_dict(data)
        return self._build_metadata_from_tensors(source, tensors_dict)

    def _load_metadata_from_directory(self, source: ModelSource) -> ModelMetadata:
        bin_files = sorted(source.path.glob("*.bin"))
        pt_files = sorted(source.path.glob("*.pt"))
        pth_files = sorted(source.path.glob("*.pth"))

        all_files = bin_files + pt_files + pth_files

        if not all_files:
            raise MalformedStateDictError(f"No PyTorch model files found in {source.path}")

        tensors_dict = {}
        for file_path in all_files:
            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                file_tensors = self._extract_state_dict(data)
                for name, tensor in file_tensors.items():
                    tensors_dict[f"{file_path.stem}.{name}"] = tensor
            except Exception as e:
                raise CorruptFileError(f"Failed to load {file_path}: {e}")

        return self._build_metadata_from_tensors(source, tensors_dict)

    def _extract_state_dict(self, data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            if "state_dict" in data:
                return data["state_dict"]
            if "model" in data and isinstance(data["model"], dict):
                return data["model"]
            return data
        raise MalformedStateDictError(f"Unexpected PyTorch file format: {type(data)}")

    def _build_loaded_model(self, source: ModelSource, tensors_dict: Dict[str, Any]) -> LoadedModel:
        config = self._load_config_from_directory(source)
        arch_info = self._extract_architecture_from_config(config) if config else {}

        tensor_metadatas = []
        loaded_tensors = {}

        for name, tensor in tensors_dict.items():
            if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
                continue

            shape = list(tensor.shape)
            dtype = self._torch_dtype_to_tensor_dtype(tensor.dtype)
            num_elements = tensor.numel()
            raw_bytes = num_elements * tensor.element_size()

            tm = TensorMetadata(
                name=name,
                shape=shape,
                dtype=dtype,
                num_elements=num_elements,
                raw_bytes=raw_bytes,
            )
            tensor_metadatas.append(tm)

            if not self.metadata_only:
                if is_numpy_available():
                    arr = tensor.detach().cpu().numpy()
                else:
                    arr = tensor.detach().cpu()
                loaded_tensors[name] = LoadedTensor(name=name, data=arr, metadata=tm)

        total_parameters = sum(tm.num_elements for tm in tensor_metadatas)

        metadata = ModelMetadata(
            model_id=source.name,
            source_path=str(source.path),
            source_format=ModelFormat.PYTORCH,
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

    def _build_metadata_from_tensors(self, source: ModelSource, tensors_dict: Dict[str, Any]) -> ModelMetadata:
        config = self._load_config_from_directory(source)
        arch_info = self._extract_architecture_from_config(config) if config else {}

        tensor_metadatas = []
        for name, tensor in tensors_dict.items():
            if not hasattr(tensor, "shape") or not hasattr(tensor, "dtype"):
                continue

            shape = list(tensor.shape)
            dtype = self._torch_dtype_to_tensor_dtype(tensor.dtype)
            num_elements = tensor.numel()
            raw_bytes = num_elements * tensor.element_size()

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
            model_id=source.name,
            source_path=str(source.path),
            source_format=ModelFormat.PYTORCH,
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

    def _torch_dtype_to_tensor_dtype(self, dtype) -> TensorDtype:
        dtype_map = {
            torch.float32: TensorDtype.FLOAT32,
            torch.float16: TensorDtype.FLOAT16,
            torch.bfloat16: TensorDtype.BFLOAT16,
            torch.int8: TensorDtype.INT8,
            torch.uint8: TensorDtype.UINT8,
            torch.bool: TensorDtype.BOOL,
        }
        return dtype_map.get(dtype, TensorDtype.UNKNOWN)

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