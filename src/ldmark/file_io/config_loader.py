from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

from .formats import ModelFormat, ModelSource
from .loaded_model import LoadedModel
from .metadata import ModelMetadata, TensorMetadata, TensorDtype
from .loader import BaseModelLoader, CorruptFileError, MalformedStateDictError


class ConfigLoader(BaseModelLoader):
    def __init__(self, metadata_only: bool = False):
        super().__init__(metadata_only)

    def can_load(self, source: ModelSource) -> bool:
        return source.format == ModelFormat.JSON_CONFIG

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
        config = self._read_config(source.path)
        return self._build_loaded_model(source, config)

    def _load_from_directory(self, source: ModelSource) -> LoadedModel:
        config_path = source.path / "config.json"
        if not config_path.exists():
            raise MalformedStateDictError(f"No config.json found in {source.path}")
        config = self._read_config(config_path)
        return self._build_loaded_model(source, config)

    def _load_metadata_from_file(self, source: ModelSource) -> ModelMetadata:
        config = self._read_config(source.path)
        return self._build_metadata(source, config)

    def _load_metadata_from_directory(self, source: ModelSource) -> ModelMetadata:
        config_path = source.path / "config.json"
        if not config_path.exists():
            raise MalformedStateDictError(f"No config.json found in {source.path}")
        config = self._read_config(config_path)
        return self._build_metadata(source, config)

    def _read_config(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise CorruptFileError(f"Invalid JSON in {path}: {e}")
        except UnicodeDecodeError as e:
            raise CorruptFileError(f"Cannot decode {path}: {e}")

    def _build_loaded_model(self, source: ModelSource, config: Dict[str, Any]) -> LoadedModel:
        arch_info = self._extract_architecture_from_config(config)
        attention_type = self._detect_attention_type(config)

        metadata = ModelMetadata(
            model_id=source.name,
            source_path=str(source.path),
            source_format=ModelFormat.JSON_CONFIG,
            architecture=arch_info.get("architecture", "unknown"),
            num_layers=arch_info.get("num_layers"),
            hidden_size=arch_info.get("hidden_size"),
            intermediate_size=arch_info.get("intermediate_size"),
            num_attention_heads=arch_info.get("num_attention_heads"),
            num_kv_heads=arch_info.get("num_kv_heads"),
            vocab_size=arch_info.get("vocab_size"),
            max_context_length=arch_info.get("max_context_length"),
            attention_type=attention_type,
            total_parameters=0,
            tensor_count=0,
            tensors=[],
            config=config,
        )

        return LoadedModel(source=source, metadata=metadata, tensors={}, config=config)

    def _build_metadata(self, source: ModelSource, config: Dict[str, Any]) -> ModelMetadata:
        arch_info = self._extract_architecture_from_config(config)
        attention_type = self._detect_attention_type(config)

        return ModelMetadata(
            model_id=source.name,
            source_path=str(source.path),
            source_format=ModelFormat.JSON_CONFIG,
            architecture=arch_info.get("architecture", "unknown"),
            num_layers=arch_info.get("num_layers"),
            hidden_size=arch_info.get("hidden_size"),
            intermediate_size=arch_info.get("intermediate_size"),
            num_attention_heads=arch_info.get("num_attention_heads"),
            num_kv_heads=arch_info.get("num_kv_heads"),
            vocab_size=arch_info.get("vocab_size"),
            max_context_length=arch_info.get("max_context_length"),
            attention_type=attention_type,
            total_parameters=0,
            tensor_count=0,
            tensors=[],
            config=config,
        )