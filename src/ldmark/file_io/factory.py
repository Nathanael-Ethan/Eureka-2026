from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .formats import ModelFormat, ModelSource, detect_format_from_path
from .loaded_model import LoadedModel
from .metadata import ModelMetadata
from .loader import ModelLoader, UnsupportedFormatError, MissingDependencyError
from .capabilities import is_torch_available, is_safetensors_available, is_numpy_available


class LoaderFactory:
    def __init__(self):
        self._loaders: Dict[ModelFormat, ModelLoader] = {}
        self._register_default_loaders()

    def _register_default_loaders(self) -> None:
        if is_numpy_available():
            from .numpy_loader import NumpyLoader
            self.register_loader(ModelFormat.NUMPY, NumpyLoader())

        if is_torch_available():
            from .torch_loader import TorchLoader
            self.register_loader(ModelFormat.PYTORCH, TorchLoader())

        if is_safetensors_available():
            from .safetensors_loader import SafetensorsLoader
            self.register_loader(ModelFormat.SAFETENSORS, SafetensorsLoader())

        from .config_loader import ConfigLoader
        self.register_loader(ModelFormat.JSON_CONFIG, ConfigLoader())

        if is_numpy_available():
            from .synthetic_loader import SyntheticLoader
            self.register_loader(ModelFormat.SYNTHETIC, SyntheticLoader())

    def register_loader(self, format: ModelFormat, loader: ModelLoader) -> None:
        self._loaders[format] = loader

    def get_loader(self, format: ModelFormat) -> ModelLoader:
        if format not in self._loaders:
            if format == ModelFormat.PYTORCH and not is_torch_available():
                raise MissingDependencyError(
                    f"Format '{format.value}' requires PyTorch. Install with: pip install torch"
                )
            if format == ModelFormat.SAFETENSORS and not is_safetensors_available():
                raise MissingDependencyError(
                    f"Format '{format.value}' requires safetensors. Install with: pip install safetensors"
                )
            raise UnsupportedFormatError(f"No loader registered for format: {format.value}")
        return self._loaders[format]

    def load(self, path: Union[str, Path], metadata_only: bool = False) -> LoadedModel:
        source = detect_format_from_path(path)
        loader = self.get_loader(source.format)

        if metadata_only:
            loader.metadata_only = True

        return loader.load(source)

    def load_metadata(self, path: Union[str, Path]) -> ModelMetadata:
        source = detect_format_from_path(path)
        loader = self.get_loader(source.format)
        return loader.load_metadata(source)

    def get_available_formats(self) -> List[str]:
        return [fmt.value for fmt in self._loaders.keys()]

    def can_load(self, path: Union[str, Path]) -> bool:
        try:
            source = detect_format_from_path(path)
            return source.format in self._loaders
        except (FileNotFoundError, UnsupportedFormatError):
            return False


_default_factory: Optional[LoaderFactory] = None


def get_default_factory() -> LoaderFactory:
    global _default_factory
    if _default_factory is None:
        _default_factory = LoaderFactory()
    return _default_factory


def load_model(path: Union[str, Path], metadata_only: bool = False) -> LoadedModel:
    return get_default_factory().load(path, metadata_only=metadata_only)


def load_model_metadata(path: Union[str, Path]) -> ModelMetadata:
    return get_default_factory().load_metadata(path)