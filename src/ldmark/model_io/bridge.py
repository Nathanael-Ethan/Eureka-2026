from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..file_io import (
    LoadedModel,
    ModelMetadata,
    ModelFormat,
    detect_format_from_path,
    load_model,
    load_model_metadata,
    get_default_factory,
)
from ..analysis import ModelAnalyzer, ModelAnalysis


class ModelIO:
    def __init__(self):
        self._factory = get_default_factory()

    def inspect(
        self,
        path: Union[str, Path],
        metadata_only: bool = True,
    ) -> ModelAnalysis:
        source = detect_format_from_path(path)

        if metadata_only:
            metadata = self._factory.load_metadata(path)
            return self._analysis_from_metadata(metadata, source)

        loaded = self._factory.load(path, metadata_only=False)
        return self._analysis_from_loaded(loaded)

    def load_and_analyze(
        self,
        path: Union[str, Path],
    ) -> ModelAnalysis:
        return self.inspect(path, metadata_only=False)

    def get_metadata(self, path: Union[str, Path]) -> ModelMetadata:
        return self._factory.load_metadata(path)

    def get_loaded_model(self, path: Union[str, Path]) -> LoadedModel:
        return self._factory.load(path, metadata_only=False)

    def _analysis_from_metadata(
        self,
        metadata: ModelMetadata,
        source,
    ) -> ModelAnalysis:
        analyzer = ModelAnalyzer(
            model_id=metadata.model_id,
            source_path=metadata.source_path,
        )

        config = metadata.config if metadata.config else {}

        from ..analysis.tensor import TensorInfo, TensorTrainableStatus
        from ..analysis.models import DType

        dtype_map = {
            "float32": DType.FP32,
            "float16": DType.FP16,
            "bfloat16": DType.BF16,
            "int8": DType.INT8,
            "int4": DType.INT4,
            "uint8": DType.INT8,
            "float8": DType.FP8,
            "float4": DType.FP4,
            "bool": DType.BIT1,
        }

        tensors = []
        for tm in metadata.tensors:
            dtype = dtype_map.get(tm.dtype.value, DType.FP16)
            tensor_info = TensorInfo(
                name=tm.name,
                shape=tm.shape,
                dtype=dtype,
                num_elements=tm.num_elements,
                estimated_raw_storage_bytes=tm.raw_bytes,
                trainable=TensorTrainableStatus.UNKNOWN,
            )
            tensors.append(tensor_info)

        analysis = analyzer.analyze_tensors(tensors, config=config)
        return analysis

    def _analysis_from_loaded(self, loaded: LoadedModel) -> ModelAnalysis:
        analyzer = ModelAnalyzer(
            model_id=loaded.model_id,
            source_path=str(loaded.source.path),
        )

        config = loaded.config if loaded.config else {}

        from ..analysis.tensor import TensorInfo, TensorTrainableStatus
        from ..analysis.models import DType
        import numpy as np

        dtype_map = {
            np.float32: DType.FP32,
            np.float16: DType.FP16,
            np.int8: DType.INT8,
            np.uint8: DType.INT8,
            np.bool_: DType.BIT1,
        }
        if hasattr(np, "bfloat16"):
            dtype_map[np.bfloat16] = DType.BF16

        tensors = []
        for name, loaded_tensor in loaded.iter_tensors():
            arr = loaded_tensor.data
            dtype = dtype_map.get(arr.dtype.type, DType.FP16)
            tensor_info = TensorInfo(
                name=name,
                shape=list(arr.shape),
                dtype=dtype,
                num_elements=arr.size,
                estimated_raw_storage_bytes=arr.size * arr.dtype.itemsize,
                trainable=TensorTrainableStatus.UNKNOWN,
            )
            tensors.append(tensor_info)

        analysis = analyzer.analyze_tensors(tensors, config=config)
        return analysis


def inspect_model(path: Union[str, Path], metadata_only: bool = True) -> ModelAnalysis:
    io = ModelIO()
    return io.inspect(path, metadata_only=metadata_only)


def load_and_analyze(path: Union[str, Path]) -> ModelAnalysis:
    io = ModelIO()
    return io.load_and_analyze(path)