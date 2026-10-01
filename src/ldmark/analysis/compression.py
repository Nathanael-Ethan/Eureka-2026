from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import DType, CompressionEstimate, ParameterCounts


@dataclass
class QuantizationFormat:
    name: str
    effective_bits_per_weight: float
    group_size: Optional[int] = None
    scale_dtype: Optional[DType] = None
    description: str = ""

    def estimate(self, param_count: int, baseline_dtype: DType = DType.FP16) -> CompressionEstimate:
        return CompressionEstimate.from_parameters(
            param_count=param_count,
            format_name=self.name,
            effective_bits_per_weight=self.effective_bits_per_weight,
            baseline_dtype=baseline_dtype,
            group_size=self.group_size,
            scale_dtype=self.scale_dtype,
            description=self.description,
        )


COMMON_FORMATS = [
    QuantizationFormat("FP32", 32.0, description="Full precision 32-bit float"),
    QuantizationFormat("FP16", 16.0, description="Half precision 16-bit float"),
    QuantizationFormat("BF16", 16.0, description="Brain float 16-bit"),
    QuantizationFormat("INT8", 8.0, description="8-bit integer quantization"),
    QuantizationFormat("INT4", 4.0, description="4-bit integer quantization"),
    QuantizationFormat("FP8", 8.0, description="8-bit float (E4M3/E5M2)"),
    QuantizationFormat("FP4", 4.0, description="4-bit float"),
    QuantizationFormat("INT2", 2.0, description="2-bit integer quantization"),
    QuantizationFormat("1-bit (naive)", 1.0, description="1-bit binary quantization, no scaling"),
    QuantizationFormat(
        "Bonsai Q1_0_g128",
        1.125,
        group_size=128,
        scale_dtype=DType.FP16,
        description="128 one-bit weights + one FP16 scale (PrismML Bonsai)",
    ),
    QuantizationFormat(
        "Q4_K_M",
        4.5,
        group_size=32,
        scale_dtype=DType.FP16,
        description="Llama.cpp Q4_K_M: 4-bit with per-block scales and mins",
    ),
    QuantizationFormat(
        "Q5_K_M",
        5.5,
        group_size=32,
        scale_dtype=DType.FP16,
        description="Llama.cpp Q5_K_M: 5-bit with per-block scales",
    ),
    QuantizationFormat(
        "Q6_K",
        6.5,
        group_size=32,
        scale_dtype=DType.FP16,
        description="Llama.cpp Q6_K: 6-bit with per-block scales",
    ),
    QuantizationFormat(
        "Q8_0",
        8.5,
        group_size=32,
        scale_dtype=DType.FP16,
        description="Llama.cpp Q8_0: 8-bit with per-block scale",
    ),
    QuantizationFormat(
        "AWQ INT4",
        4.125,
        group_size=128,
        scale_dtype=DType.FP16,
        description="AWQ 4-bit with per-group FP16 scales and zeros",
    ),
    QuantizationFormat(
        "GPTQ INT4",
        4.125,
        group_size=128,
        scale_dtype=DType.FP16,
        description="GPTQ 4-bit with per-group FP16 scales",
    ),
    QuantizationFormat(
        "FP8 (per-tensor)",
        8.0,
        description="FP8 E4M3 with single scale per tensor",
    ),
    QuantizationFormat(
        "FP8 (per-channel)",
        8.02,
        group_size=128,
        scale_dtype=DType.FP32,
        description="FP8 with per-channel FP32 scales",
    ),
]


def get_format_by_name(name: str) -> Optional[QuantizationFormat]:
    for fmt in COMMON_FORMATS:
        if fmt.name.lower() == name.lower():
            return fmt
    return None


def estimate_compression_for_format(
    param_count: int,
    format_name: str,
    baseline_dtype: DType = DType.FP16,
) -> Optional[CompressionEstimate]:
    fmt = get_format_by_name(format_name)
    if fmt:
        return fmt.estimate(param_count, baseline_dtype)
    return None


def estimate_all_compressions(
    param_count: int,
    baseline_dtype: DType = DType.FP16,
    formats: Optional[List[QuantizationFormat]] = None,
) -> List[CompressionEstimate]:
    formats = formats or COMMON_FORMATS
    return [fmt.estimate(param_count, baseline_dtype) for fmt in formats]


def estimate_compression_from_parameter_counts(
    counts: ParameterCounts,
    baseline_dtype: DType = DType.FP16,
) -> List[CompressionEstimate]:
    return estimate_all_compressions(counts.total, baseline_dtype)


def calculate_effective_bps(
    weight_bits: int,
    group_size: int,
    scale_bits: int = 16,
    zero_point_bits: int = 0,
) -> float:
    if group_size <= 1:
        return float(weight_bits)
    scale_overhead = scale_bits / group_size
    zero_overhead = zero_point_bits / group_size if zero_point_bits > 0 else 0
    return weight_bits + scale_overhead + zero_overhead


def create_custom_format(
    name: str,
    weight_bits: int,
    group_size: Optional[int] = None,
    scale_dtype: DType = DType.FP16,
    zero_point: bool = False,
    description: str = "",
) -> QuantizationFormat:
    scale_bits = scale_dtype.bits
    zero_bits = scale_dtype.bits if zero_point else 0
    effective_bps = calculate_effective_bps(weight_bits, group_size or 0, scale_bits, zero_bits)
    return QuantizationFormat(
        name=name,
        effective_bits_per_weight=effective_bps,
        group_size=group_size,
        scale_dtype=scale_dtype,
        description=description or f"{weight_bits}-bit weights, group={group_size}, scale={scale_dtype.value}",
    )


def get_compression_summary(estimates: List[CompressionEstimate]) -> Dict[str, Any]:
    if not estimates:
        return {}

    sorted_estimates = sorted(estimates, key=lambda x: x.effective_bits_per_weight)
    return {
        "formats": [e.to_dict() for e in sorted_estimates],
        "best_compression": sorted_estimates[0].format_name if sorted_estimates else None,
        "smallest_size_gb": sorted_estimates[0].estimated_gb if sorted_estimates else None,
    }